<script lang="ts">
	import '$lib/stats.css';
	import { getBehaviorStats, type BehaviorReport } from '$lib/api';
	import BehaviorTable, { SIGNALS } from '$lib/BehaviorTable.svelte';
	import ScrollableTable from '$lib/ScrollableTable.svelte';

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

<section class="statistics behavior">
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
				><option value="claude">Claude</option><option value="omp">OMP</option><option value="pi"
					>Pi</option
				>
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
				<ScrollableTable>
					<BehaviorTable rows={report.models} field="model" label="Model" code />
				</ScrollableTable>
			</section>
			<section>
				<h2>By agent</h2>
				<ScrollableTable>
					<BehaviorTable rows={report.agents} field="agent" label="Agent" />
				</ScrollableTable>
			</section>
			<section>
				<h2>By project</h2>
				<ScrollableTable>
					<BehaviorTable rows={report.projects} field="cwd" label="Project" code />
				</ScrollableTable>
			</section>
			<section>
				<h2>Daily trend</h2>
				<ScrollableTable>
					<BehaviorTable rows={report.daily} field="date" label="Date" />
				</ScrollableTable>
			</section>
		</div>
	{/if}
</section>

<style>
	.context-strip {
		display: grid;
		gap: 1px;
		grid-template-columns: repeat(4, 1fr);
		overflow: hidden;
		border: 1px solid var(--border);
		border-radius: var(--r3);
		background: var(--border);
	}
	.charts {
		align-items: start;
	}
	p {
		margin: 0;
	}
	h2 {
		margin: 0 0 var(--s3);
	}
	.table-note {
		font-size: 12px;
		margin-bottom: var(--s2);
	}
	.card strong {
		font-size: 21px;
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
	.chart-card {
		align-self: start;
	}
	.section-heading h2 {
		margin-bottom: var(--s2);
	}
	.bar-label {
		justify-content: space-between;
		font-size: 11px;
	}
	.bar-meta {
		display: flex;
		justify-content: space-between;
		gap: var(--s2);
		min-width: 0;
		font-size: 10px;
		color: var(--muted);
	}
	.bar-label span {
		color: var(--muted);
		font-variant-numeric: tabular-nums;
	}
	.track {
		height: 8px;
	}
	.bar {
		background: linear-gradient(90deg, var(--accent), #a78bfa);
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
		color: var(--muted);
	}
	@media (max-width: 640px) {
		.context-strip {
			grid-template-columns: repeat(2, 1fr);
		}
	}
</style>
