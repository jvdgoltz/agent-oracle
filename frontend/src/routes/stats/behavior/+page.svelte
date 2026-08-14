<script lang="ts">
	import { getBehaviorStats, type BehaviorReport, type BehaviorSummary } from '$lib/api';

	/** Signals exposed by OMP's user-metrics classifier. */
	const SIGNALS: {
		key: keyof BehaviorSummary;
		rate: keyof BehaviorSummary;
		label: string;
		unit: string;
	}[] = [
		{ key: 'yelling', rate: 'yelling_rate', label: 'Yelling', unit: 'hits' },
		{ key: 'profanity', rate: 'profanity_rate', label: 'Profanity', unit: 'hits' },
		{ key: 'anguish', rate: 'anguish_rate', label: 'Anguish', unit: 'hits' },
		{ key: 'negation', rate: 'negation_rate', label: 'Negation', unit: 'hits' },
		{ key: 'repetition', rate: 'repetition_rate', label: 'Repetition', unit: 'hits' },
		{ key: 'blame', rate: 'blame_rate', label: 'Blame', unit: 'hits' },
		{
			key: 'interruptions',
			rate: 'interruption_rate',
			label: 'Interruptions',
			unit: 'interruptions'
		}
	];

	let agent = $state('');
	let start = $state('');
	let end = $state('');
	let report = $state<BehaviorReport | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	/** Format an OMP per-message rate for display. */
	function rate(value: number): string {
		return `${value.toFixed(2)}%`;
	}

	/** Keep chart bars visible while preserving zero values as zero. */
	function barWidth(value: number, maximum: number): string {
		return `${maximum > 0 ? Math.max(value > 0 ? 4 : 0, (value / maximum) * 100) : 0}%`;
	}

	/** Format dates compactly for the trend axis. */
	function shortDate(value: string): string {
		const date = new Date(`${value}T00:00:00`);
		return Number.isNaN(date.valueOf())
			? value
			: date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
	}

	/** Round a positive rate up to a compact, truthful chart ceiling. */
	function trendMaximum(rows: BehaviorReport['daily']): number {
		const maximum = Math.max(0, ...rows.map((row) => row.detection_rate));
		if (maximum === 0) return 1;
		const magnitude = 10 ** Math.floor(Math.log10(maximum));
		const normalized = maximum / magnitude;
		const step = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
		return step * magnitude;
	}

	/** Format chart-axis rates without implying a fixed 100% scale. */
	function trendLabel(value: number, maximum: number): string {
		const decimals = maximum < 1 ? 2 : maximum < 10 ? 1 : 0;
		return `${value.toFixed(decimals)}%`;
	}

	/** Map a daily rate to the shared chart plot area. */
	function trendY(value: number, maximum: number): number {
		return 92 - (value / maximum) * 84;
	}

	/** Build an accessible SVG polyline for the daily detection-rate trend. */
	function trendPoints(rows: BehaviorReport['daily'], maximum: number): string {
		if (!rows.length) return '';
		return rows
			.map((row, index) => {
				const x = rows.length === 1 ? 50 : (index / (rows.length - 1)) * 100;
				return `${x},${trendY(row.detection_rate, maximum)}`;
			})
			.join(' ');
	}

	/** Keep the trend readable by exposing only a small, evenly spaced marker set. */
	function trendMarkerIndexes(rows: BehaviorReport['daily']): number[] {
		const count = Math.min(rows.length, 8);
		if (count <= 1) return rows.length ? [0] : [];
		return Array.from({ length: count }, (_, index) =>
			Math.round((index * (rows.length - 1)) / (count - 1))
		);
	}

	/** Load behavior statistics for the current bounded filters. */
	async function loadStats() {
		loading = true;
		error = null;
		try {
			report = await getBehaviorStats(agent || undefined, start || undefined, end || undefined);
		} catch (exception) {
			error = exception instanceof Error ? exception.message : 'Failed to load behavior statistics';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		loadStats();
	});
</script>

<section class="behavior">
	<div class="heading">
		<div>
			<h1>User behavior</h1>
			<p>OMP-compatible signals from real user messages across the archive.</p>
		</div>
		<span class="scope">{report?.totals.user_messages ?? 0} user messages</span>
	</div>

	<form
		class="filters"
		onsubmit={(event: SubmitEvent) => {
			event.preventDefault();
			loadStats();
		}}
	>
		<label
			>Agent
			<select bind:value={agent}>
				<option value="">All agents</option><option value="codex">Codex</option><option
					value="factory">Factory</option
				><option value="claude">Claude</option><option value="omp">OMP</option>
			</select>
		</label>
		<label>From <input type="date" bind:value={start} /></label>
		<label>To <input type="date" bind:value={end} /></label>
	</form>

	{#if loading}
		<p class="status">Loading behavior statistics…</p>
	{:else if error}
		<p class="error">{error}</p>
	{:else if report}
		{@const trendRows = [...report.daily].sort((left, right) =>
			left.date.localeCompare(right.date)
		)}
		<div class="cards">
			{#each SIGNALS as signal (signal.key)}
				<div class="card">
					<span>{signal.label}</span><strong>{rate(report.totals[signal.rate] as number)}</strong
					><small>{report.totals[signal.key] as number} {signal.unit}</small>
				</div>
			{/each}
		</div>

		<div class="context-strip">
			<div><span>Messages analyzed</span><strong>{report.totals.user_messages}</strong></div>
			<div>
				<span>Interruptions</span><strong>{report.totals.interruptions}</strong><small
					>source-recorded stops</small
				>
			</div>
			<div>
				<span>Detected messages</span><strong>{report.totals.detected_messages}</strong><small
					>{rate(report.totals.detection_rate)} of user messages</small
				>
			</div>
			<div>
				<span>Signal hits</span><strong
					>{report.totals.yelling +
						report.totals.profanity +
						report.totals.anguish +
						report.totals.negation +
						report.totals.repetition +
						report.totals.blame}</strong
				><small>across six signals</small>
			</div>
		</div>

		<div class="charts">
			<section class="chart-card">
				<div class="section-heading">
					<div>
						<h2>Detection by model</h2>
						<p>Detected user messages divided by messages analyzed.</p>
					</div>
					<span class="chart-total">{rate(report.totals.detection_rate)} overall</span>
				</div>
				{#if report.models.length}
					<div class="bar-list">
						{#each report.models.slice(0, 8) as row (row.model)}
							<div
								class="bar-row"
								title={`${row.model}: ${rate(row.detection_rate)} detection rate, ${row.detected_messages} detected messages`}
							>
								<div class="bar-label">
									<code>{row.model}</code><span>{row.detected_messages}/{row.user_messages}</span>
								</div>
								<div class="track">
									<span
										class="bar"
										style={`width: ${barWidth(row.detection_rate, Math.max(...report.models.map((item) => item.detection_rate)))}`}
									></span>
								</div>
								<div class="bar-meta">
									<span>{rate(row.detection_rate)} detection rate</span><span
										>{rate(row.interruption_rate)} interruptions</span
									>
								</div>
							</div>
						{/each}
					</div>
				{:else}<p class="empty">No model data in this scope.</p>{/if}
			</section>
			<section class="chart-card">
				<div class="section-heading">
					<div>
						<h2>Daily detection trend</h2>
						<p>Rate is normalized per day, so quiet days stay comparable.</p>
					</div>
					<span class="chart-total">{trendRows.length} days</span>
				</div>
				{#if trendRows.length}
					{@const maximum = trendMaximum(trendRows)}
					{@const points = trendPoints(trendRows, maximum)}
					{@const markerIndexes = trendMarkerIndexes(trendRows)}
					<div class="trend-wrap">
						<div class="trend-plot">
							<div class="trend-y-axis" aria-hidden="true">
								<span>{trendLabel(maximum, maximum)}</span>
								<span>{trendLabel(maximum / 2, maximum)}</span>
								<span>0%</span>
							</div>
							<div class="trend-canvas">
								<svg
									class="trend"
									viewBox="0 0 100 100"
									preserveAspectRatio="none"
									role="img"
									aria-label={`Daily detection rate trend from ${shortDate(trendRows[0].date)} to ${shortDate(trendRows[trendRows.length - 1].date)}, with a maximum scale of ${trendLabel(maximum, maximum)}`}
								>
									<line x1="0" y1="8" x2="100" y2="8" />
									<line x1="0" y1="50" x2="100" y2="50" />
									<line x1="0" y1="92" x2="100" y2="92" />
									<polyline {points} />
								</svg>
								{#each trendRows as row, index (row.date)}
									{@const x = trendRows.length === 1 ? 50 : (index / (trendRows.length - 1)) * 100}
									{#if markerIndexes.includes(index)}
										<button
											class="trend-marker"
											type="button"
											style={`left: ${x}%; top: ${trendY(row.detection_rate, maximum)}%`}
											aria-label={`${shortDate(row.date)}: ${rate(row.detection_rate)}, ${row.detected_messages} detected`}
											title={`${shortDate(row.date)}: ${rate(row.detection_rate)} (${row.detected_messages} detected)`}
										></button>
									{/if}
								{/each}
							</div>
						</div>
						<div class="trend-x-axis">
							<span>{shortDate(trendRows[0].date)}</span>
							<span>{shortDate(trendRows[trendRows.length - 1].date)}</span>
						</div>
						<details class="trend-data">
							<summary>View exact daily values</summary>
							<ul>
								{#each trendRows as row (row.date)}
									<li>
										<span>{shortDate(row.date)}</span>
										<strong>{rate(row.detection_rate)}</strong>
										<span>{row.detected_messages} detected</span>
									</li>
								{/each}
							</ul>
						</details>
					</div>
				{:else}<p class="empty">No daily data in this scope.</p>{/if}
			</section>
		</div>

		<div class="tables">
			<section>
				<h2>By model</h2>
				<p class="table-note">
					Normal rows use the previous eligible assistant response; interruptions use the
					source-recorded model.
				</p>
				<div class="table-scroll">
					<table>
						<thead
							><tr
								><th>Model</th><th>Messages</th><th>Yelling</th><th>Profanity</th><th>Anguish</th
								><th>Negation</th><th>Repetition</th><th>Blame</th><th>Interruptions</th><th
									>Total</th
								><th class="percentage">Percentage</th></tr
							></thead
						><tbody>
							{#each report.models as row (row.model)}<tr
									><td><code>{row.model}</code></td><td>{row.user_messages}</td><td
										>{rate(row.yelling_rate)}</td
									><td>{rate(row.profanity_rate)}</td><td>{rate(row.anguish_rate)}</td><td
										>{rate(row.negation_rate)}</td
									><td>{rate(row.repetition_rate)}</td><td>{rate(row.blame_rate)}</td><td
										>{rate(row.interruption_rate)}</td
									><td>{row.detected_messages}</td><td class="percentage"
										>{rate(row.detection_rate)}</td
									></tr
								>{/each}
						</tbody>
					</table>
				</div>
			</section>
			<section>
				<h2>By agent</h2>
				<div class="table-scroll">
					<table>
						<thead
							><tr
								><th>Agent</th><th>Messages</th><th>Yelling</th><th>Profanity</th><th>Anguish</th
								><th>Negation</th><th>Repetition</th><th>Blame</th><th>Interruptions</th><th
									>Total</th
								><th class="percentage">Percentage</th></tr
							></thead
						><tbody>
							{#each report.agents as row (row.agent)}<tr
									><td>{row.agent}</td><td>{row.user_messages}</td><td>{rate(row.yelling_rate)}</td
									><td>{rate(row.profanity_rate)}</td><td>{rate(row.anguish_rate)}</td><td
										>{rate(row.negation_rate)}</td
									><td>{rate(row.repetition_rate)}</td><td>{rate(row.blame_rate)}</td><td
										>{rate(row.interruption_rate)}</td
									><td>{row.detected_messages}</td><td class="percentage"
										>{rate(row.detection_rate)}</td
									></tr
								>{/each}
						</tbody>
					</table>
				</div>
			</section>
			<section>
				<h2>By project</h2>
				<div class="table-scroll">
					<table>
						<thead
							><tr
								><th>Project</th><th>Messages</th><th>Yelling</th><th>Profanity</th><th>Anguish</th
								><th>Negation</th><th>Repetition</th><th>Blame</th><th>Interruptions</th><th
									>Total</th
								><th class="percentage">Percentage</th></tr
							></thead
						><tbody>
							{#each report.projects as row (row.cwd)}<tr
									><td><code>{row.cwd}</code></td><td>{row.user_messages}</td><td
										>{rate(row.yelling_rate)}</td
									><td>{rate(row.profanity_rate)}</td><td>{rate(row.anguish_rate)}</td><td
										>{rate(row.negation_rate)}</td
									><td>{rate(row.repetition_rate)}</td><td>{rate(row.blame_rate)}</td><td
										>{rate(row.interruption_rate)}</td
									><td>{row.detected_messages}</td><td class="percentage"
										>{rate(row.detection_rate)}</td
									></tr
								>{/each}
						</tbody>
					</table>
				</div>
			</section>
			<section>
				<h2>Daily trend</h2>
				<div class="table-scroll">
					<table>
						<thead
							><tr
								><th>Date</th><th>Messages</th><th>Yelling</th><th>Profanity</th><th>Anguish</th><th
									>Negation</th
								><th>Repetition</th><th>Blame</th><th>Interruptions</th><th>Total</th><th
									class="percentage">Percentage</th
								></tr
							></thead
						>
						<tbody>
							{#each report.daily as row (row.date)}
								<tr
									><td>{row.date}</td><td>{row.user_messages}</td><td>{rate(row.yelling_rate)}</td
									><td>{rate(row.profanity_rate)}</td><td>{rate(row.anguish_rate)}</td><td
										>{rate(row.negation_rate)}</td
									><td>{rate(row.repetition_rate)}</td><td>{rate(row.blame_rate)}</td><td
										>{rate(row.interruption_rate)}</td
									><td>{row.detected_messages}</td><td class="percentage"
										>{rate(row.detection_rate)}</td
									></tr
								>
							{/each}
						</tbody>
					</table>
				</div>
			</section>
		</div>
	{/if}
</section>

<style>
	.behavior {
		display: grid;
		gap: var(--s5);
	}
	.heading,
	.filters,
	.cards,
	.context-strip,
	.charts,
	.tables {
		display: flex;
		gap: var(--s4);
	}
	.heading {
		align-items: end;
		justify-content: space-between;
	}
	h1,
	h2,
	p {
		margin: 0;
	}
	h1 {
		font-size: 22px;
	}
	h2 {
		font-size: 15px;
		margin-bottom: var(--s3);
	}
	p,
	small,
	.scope {
		color: var(--muted);
	}
	.table-note {
		font-size: 12px;
		margin-bottom: var(--s2);
	}
	.filters {
		flex-wrap: wrap;
	}
	label {
		display: grid;
		gap: var(--s1);
		color: var(--muted);
		font-size: 12px;
	}
	select,
	input {
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--r2);
		color: var(--text);
		padding: var(--s2);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
	}
	.card {
		display: grid;
		gap: var(--s1);
		padding: var(--s4);
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--r3);
	}
	.card strong {
		font-size: 21px;
		font-variant-numeric: tabular-nums;
	}
	.card small {
		font-size: 11px;
	}
	.context-strip {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 1px;
		overflow: hidden;
		border: 1px solid var(--border);
		border-radius: var(--r3);
		background: var(--border);
	}
	.context-strip > div {
		display: grid;
		gap: var(--s1);
		padding: var(--s3) var(--s4);
		background: var(--surface);
	}
	.context-strip span,
	.context-strip small {
		color: var(--muted);
		font-size: 11px;
	}
	.context-strip strong {
		font-size: 18px;
		font-variant-numeric: tabular-nums;
	}
	.charts {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		align-items: start;
	}
	.chart-card {
		min-width: 0;
		align-self: start;
		padding: var(--s4);
		background: linear-gradient(160deg, var(--surface), var(--bg));
		border: 1px solid var(--border);
		border-radius: var(--r3);
	}
	.section-heading {
		display: flex;
		align-items: start;
		justify-content: space-between;
		gap: var(--s3);
		margin-bottom: var(--s4);
	}
	.section-heading h2 {
		margin-bottom: var(--s2);
	}
	.section-heading p {
		margin: 0;
		font-size: 11px;
	}
	.chart-total {
		color: var(--accent);
		font-size: 11px;
		white-space: nowrap;
	}
	.bar-list {
		display: grid;
		gap: var(--s3);
	}
	.bar-row {
		display: grid;
		gap: var(--s1);
	}
	.bar-label,
	.bar-meta {
		display: flex;
		justify-content: space-between;
		gap: var(--s2);
		min-width: 0;
		font-size: 11px;
	}
	.bar-label span {
		color: var(--muted);
		font-variant-numeric: tabular-nums;
	}
	.bar-label code {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.track {
		height: 8px;
		overflow: hidden;
		border-radius: 99px;
		background: var(--elevated);
	}
	.bar {
		display: block;
		height: 100%;
		min-width: 0;
		border-radius: inherit;
		background: linear-gradient(90deg, var(--accent), #a78bfa);
	}
	.bar-meta {
		color: var(--muted);
		font-size: 10px;
	}
	.trend-wrap {
		display: grid;
		gap: var(--s2);
	}
	.trend-plot {
		display: grid;
		grid-template-columns: 36px minmax(0, 1fr);
		align-items: stretch;
		gap: var(--s2);
	}
	.trend {
		width: 100%;
		height: 168px;
		display: block;
		overflow: visible;
	}
	.trend-canvas {
		position: relative;
		min-width: 0;
	}
	.trend line {
		stroke: var(--border);
		stroke-width: 0.6;
		stroke-dasharray: 2 2;
	}
	.trend polyline {
		fill: none;
		stroke: var(--accent);
		stroke-linecap: round;
		stroke-linejoin: round;
		stroke-width: 2;
		vector-effect: non-scaling-stroke;
	}
	.trend-marker {
		position: absolute;
		width: 7px;
		height: 7px;
		padding: 0;
		border: 1.5px solid var(--accent);
		border-radius: 50%;
		background: var(--bg);
		cursor: help;
		transform: translate(-50%, -50%);
	}
	.trend-marker:hover,
	.trend-marker:focus-visible {
		background: var(--accent);
		box-shadow: 0 0 0 3px var(--accent-glow);
		outline: none;
	}
	.trend-y-axis {
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		padding: 0 0 var(--s1);
		color: var(--muted);
		font-size: 10px;
		text-align: right;
	}
	.trend-x-axis {
		display: flex;
		justify-content: space-between;
		color: var(--muted);
		font-size: 10px;
	}
	.trend-data {
		color: var(--muted);
		font-size: 10px;
	}
	.trend-data summary {
		cursor: pointer;
		width: fit-content;
	}
	.trend-data ul {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
		gap: var(--s1) var(--s3);
		max-height: 160px;
		overflow-y: auto;
		margin: var(--s2) 0 0;
		padding: 0;
		list-style: none;
	}
	.trend-data li {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 0 var(--s2);
	}
	.trend-data li span:last-child {
		grid-column: 1 / -1;
		font-size: 9px;
	}
	.trend-data strong {
		color: var(--text);
		font-variant-numeric: tabular-nums;
	}
	.empty {
		padding: var(--s5) 0;
		color: var(--muted);
		text-align: center;
	}
	.tables {
		align-items: start;
		flex-wrap: wrap;
	}
	.tables section {
		flex: 1 1 350px;
	}
	table {
		width: 100%;
		border-collapse: collapse;
	}
	.table-scroll {
		overflow-x: auto;
	}
	th,
	td {
		padding: var(--s2);
		border-bottom: 1px solid var(--border);
		text-align: left;
	}
	th {
		color: var(--muted);
		font-size: 12px;
		font-weight: 500;
	}
	.percentage {
		font-weight: 700;
	}
	@media (max-width: 640px) {
		.context-strip {
			grid-template-columns: repeat(2, 1fr);
		}
		.charts {
			grid-template-columns: 1fr;
		}
		.heading {
			align-items: start;
			flex-direction: column;
		}
	}
</style>
