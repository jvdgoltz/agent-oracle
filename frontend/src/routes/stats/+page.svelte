<script lang="ts">
	import {
		getOverviewStats,
		getTokenUsageStats,
		type OverviewReport,
		type TokenUsageReport
	} from '$lib/api';
	import ScrollableTable from '$lib/ScrollableTable.svelte';

	let agent = $state('');
	let start = $state('');
	let end = $state('');
	let report = $state<OverviewReport | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let tokenReport = $state<TokenUsageReport | null>(null);

	const agentColors: Record<string, string> = {
		codex: 'var(--codex-color)',
		factory: 'var(--factory-color)',
		claude: 'var(--claude-color)',
		omp: 'var(--omp-color)',
		pi: '#c0a0ff'
	};

	/** Format an elapsed duration in concise human-readable units. */
	function duration(seconds: number): string {
		if (seconds < 60) return `${seconds.toFixed(0)}s`;
		if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
		return `${(seconds / 3600).toFixed(1)}h`;
	}

	/** Load overview statistics for the current bounded filters. */
	async function loadStats() {
		loading = true;
		error = null;
		try {
			const filters = [agent || undefined, start || undefined, end || undefined] as const;
			[report, tokenReport] = await Promise.all([
				getOverviewStats(...filters),
				getTokenUsageStats(...filters)
			]);
		} catch (exception) {
			error = exception instanceof Error ? exception.message : 'Failed to load overview statistics';
		} finally {
			loading = false;
		}
	}

	/** Return a compact count for chart labels without hiding exact values in titles. */
	function compact(value: number): string {
		return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(
			value
		);
	}

	/** Format a provider token count in millions while preserving unavailable values. */
	function tokenMillions(value: number | null): string {
		return value === null ? '—' : `${(value / 1_000_000).toFixed(1)}M`;
	}

	/** Return a safe percentage for a horizontal bar. */
	function width(value: number, maximum: number): string {
		return `${maximum > 0 && value > 0 ? Math.max(3, (value / maximum) * 100) : 0}%`;
	}

	/** Group archived sessions into useful message-count bands. */
	function lengthBuckets(report: OverviewReport) {
		const buckets = [
			{ label: '1–4', min: 1, max: 4, count: 0 },
			{ label: '5–9', min: 5, max: 9, count: 0 },
			{ label: '10–19', min: 10, max: 19, count: 0 },
			{ label: '20–39', min: 20, max: 39, count: 0 },
			{ label: '40+', min: 40, max: Number.POSITIVE_INFINITY, count: 0 }
		];
		for (const session of report.session_lengths) {
			const bucket = buckets.find(
				(candidate) => session.messages >= candidate.min && session.messages <= candidate.max
			);
			if (bucket) bucket.count += 1;
		}
		return buckets;
	}

	$effect(() => {
		loadStats();
	});
</script>

<section class="overview">
	<div class="heading">
		<div>
			<h1>Archive overview</h1>
			<p>Sessions, conversation messages, models, and entities.</p>
		</div>
		<span class="scope">{report?.totals.sessions ?? 0} sessions</span>
	</div>

	<form
		class="filters"
		onsubmit={(event: SubmitEvent) => {
			event.preventDefault();
			loadStats();
		}}
	>
		<label
			>Agent <select bind:value={agent}
				><option value="">All agents</option><option value="codex">Codex</option><option
					value="factory">Factory</option
				><option value="claude">Claude</option><option value="omp">OMP</option><option value="pi"
					>Pi</option
				></select
			></label
		>
		<label>From <input type="date" bind:value={start} /></label>
		<label>To <input type="date" bind:value={end} /></label>
	</form>

	{#if loading}
		<p class="status">Loading archive statistics…</p>
	{:else if error}
		<p class="error">{error}</p>
	{:else if report}
		<div class="cards">
			<div class="card accent">
				<span>Sessions</span><strong>{report.totals.sessions}</strong><small>archived threads</small
				>
			</div>
			<div class="card">
				<span>Conversation messages</span><strong>{report.totals.conversation_messages}</strong>
				<small>{report.totals.average_session_messages.toFixed(1)} per session</small>
			</div>
			<div class="card">
				<span>Assistant messages</span><strong>{report.totals.assistant_messages}</strong><small
					>responses in archive</small
				>
			</div>
			<div class="card">
				<span>Median session length</span><strong
					>{report.totals.median_session_messages.toFixed(1)} messages</strong
				><small>{duration(report.totals.median_session_duration_seconds)} elapsed</small>
			</div>
		</div>

		{#if tokenReport}
			<section class="chart-card wide-chart token-usage">
				<div class="section-heading">
					<div>
						<h2>Token consumption</h2>
						<p>Provider-reported usage; unavailable fields remain blank.</p>
					</div>
				</div>
				<ScrollableTable
					><table>
						<thead
							><tr
								><th>Agent</th><th>Model</th><th>Responses</th><th>Input</th><th>Output</th><th
									>Total</th
								></tr
							></thead
						>
						<tbody
							>{#each tokenReport.agent_model as row (`${row.agent}:${row.model}`)}
								<tr
									><td>{row.agent}</td><td><code>{row.model}</code></td><td>{row.responses}</td><td
										>{tokenMillions(row.input_tokens)}</td
									><td>{tokenMillions(row.output_tokens)}</td><td
										>{tokenMillions(row.total_tokens)}</td
									></tr
								>
							{/each}</tbody
						>
					</table></ScrollableTable
				>
			</section>
		{/if}

		<div class="charts">
			<section class="chart-card">
				<div class="section-heading">
					<div>
						<h2>Sessions by agent</h2>
						<p>Which tools contribute most of the archive?</p>
					</div>
					<span class="chart-total">{compact(report.totals.sessions)} total</span>
				</div>
				{#if report.agents.length}
					<div class="bar-list">
						{#each report.agents as row (row.agent)}
							<div class="bar-row" title={`${row.agent}: ${row.sessions} sessions`}>
								<div class="bar-label">
									<span
										class="dot"
										style={`--dot-color: ${agentColors[row.agent] ?? 'var(--accent)'}`}
									></span><span>{row.agent}</span><strong>{row.sessions}</strong>
								</div>
								<div class="track">
									<span
										class="bar"
										style={`width: ${width(row.sessions, Math.max(...report.agents.map((item) => item.sessions)))}; background: ${agentColors[row.agent] ?? 'var(--accent)'}`}
									></span>
								</div>
							</div>
						{/each}
					</div>
				{:else}<p class="empty">No sessions in this scope.</p>{/if}
			</section>
			<section class="chart-card">
				<div class="section-heading">
					<div>
						<h2>Assistant messages by model</h2>
						<p>Model volume across assistant messages.</p>
					</div>
					<span class="chart-total">{compact(report.totals.assistant_messages)} total</span>
				</div>
				{#if report.models.length}
					<div class="bar-list">
						{#each report.models.slice(0, 8) as row (row.model)}
							<div class="bar-row" title={`${row.model}: ${row.messages} assistant messages`}>
								<div class="bar-label"><code>{row.model}</code><strong>{row.messages}</strong></div>
								<div class="track">
									<span
										class="bar model-bar"
										style={`width: ${width(row.messages, Math.max(...report.models.map((item) => item.messages)))}`}
									></span>
								</div>
							</div>
						{/each}
					</div>
				{:else}<p class="empty">No model data in this scope.</p>{/if}
			</section>
			<section class="chart-card wide-chart">
				<div class="section-heading">
					<div>
						<h2>Session length distribution</h2>
						<p>Sessions grouped by visible user and assistant messages.</p>
					</div>
					<span class="chart-total">{report.session_lengths.length} measured</span>
				</div>
				{#if report.session_lengths.length}
					<div class="histogram" aria-label="Session length distribution">
						{#each lengthBuckets(report) as bucket (bucket.label)}
							<div
								class="histogram-column"
								title={`${bucket.count} sessions with ${bucket.label} messages`}
							>
								<strong>{bucket.count}</strong>
								<div class="histogram-track">
									<span
										style={`height: ${width(bucket.count, Math.max(...lengthBuckets(report).map((item) => item.count)))}`}
									></span>
								</div>
								<span>{bucket.label}</span>
							</div>
						{/each}
					</div>
				{:else}<p class="empty">No session length data in this scope.</p>{/if}
			</section>
		</div>

		<div class="tables">
			<section>
				<h2>Sessions by agent</h2>
				<ScrollableTable
					><table>
						<thead><tr><th>Agent</th><th>Sessions</th></tr></thead><tbody
							>{#each report.agents as row (row.agent)}<tr
									><td>{row.agent}</td><td>{row.sessions}</td></tr
								>{/each}</tbody
						>
					</table></ScrollableTable
				>
			</section>
			<section>
				<h2>Sessions by project</h2>
				<ScrollableTable
					><table>
						<thead><tr><th>Project</th><th>Sessions</th></tr></thead><tbody
							>{#each report.projects as row (row.cwd)}<tr
									><td><code>{row.cwd}</code></td><td>{row.sessions}</td></tr
								>{/each}</tbody
						>
					</table></ScrollableTable
				>
			</section>
			<section>
				<h2>Messages by model</h2>
				<ScrollableTable
					><table>
						<thead><tr><th>Model</th><th>Messages</th></tr></thead><tbody
							>{#each report.models as row (row.model)}<tr
									><td><code>{row.model}</code></td><td>{row.messages}</td></tr
								>{/each}</tbody
						>
					</table></ScrollableTable
				>
			</section>
			<section>
				<h2>Sessions by entity</h2>
				<ScrollableTable>
					<table>
						<thead><tr><th>Type</th><th>Entity</th><th>Sessions</th></tr></thead><tbody
							>{#each report.entities as row (`${row.entity_type}:${row.entity_value}`)}<tr
									><td>{row.entity_type}</td><td><code>{row.entity_value}</code></td><td
										>{row.sessions}</td
									></tr
								>{/each}</tbody
						>
					</table>
				</ScrollableTable>
			</section>
			<section class="wide">
				<h2>Session length</h2>
				<p>Visible user and assistant messages; elapsed from first to last message.</p>
				<ScrollableTable>
					<table>
						<thead
							><tr
								><th>Session</th><th>Agent</th><th>Project</th><th>Messages</th><th>Elapsed</th></tr
							></thead
						><tbody
							>{#each report.session_lengths as row (row.session_id)}<tr
									><td><code>{row.session_id}</code></td><td>{row.agent}</td><td
										><code>{row.cwd}</code></td
									><td>{row.messages}</td><td>{duration(row.duration_seconds)}</td></tr
								>{/each}</tbody
						>
					</table>
				</ScrollableTable>
			</section>
		</div>
	{/if}
</section>

<style>
	.overview {
		display: grid;
		gap: var(--s5);
	}
	.heading,
	.filters,
	.cards,
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
		margin-bottom: var(--s2);
	}
	p,
	small,
	.scope {
		color: var(--muted);
	}
	p {
		font-size: 12px;
		margin-bottom: var(--s2);
	}
	.filters,
	.tables {
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
		box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12);
	}
	.card.accent {
		border-color: color-mix(in srgb, var(--accent) 45%, var(--border));
		background: linear-gradient(145deg, var(--accent-dim), var(--surface) 70%);
	}
	.card strong {
		font-size: 20px;
		font-variant-numeric: tabular-nums;
	}
	.card small {
		font-size: 11px;
	}
	.charts {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}
	.chart-card {
		min-width: 0;
		padding: var(--s4);
		background: linear-gradient(160deg, var(--surface), var(--bg));
		border: 1px solid var(--border);
		border-radius: var(--r3);
	}
	.wide-chart {
		grid-column: 1 / -1;
	}
	.section-heading {
		display: flex;
		align-items: start;
		justify-content: space-between;
		gap: var(--s3);
		margin-bottom: var(--s4);
	}
	.section-heading p {
		margin: 0;
		font-size: 11px;
	}
	.chart-total {
		color: var(--accent);
		font-size: 11px;
		font-variant-numeric: tabular-nums;
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
	.bar-label {
		display: flex;
		align-items: center;
		gap: var(--s2);
		min-width: 0;
		font-size: 12px;
	}
	.bar-label span:not(.dot),
	.bar-label code {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.bar-label strong {
		margin-left: auto;
		font-size: 12px;
		font-variant-numeric: tabular-nums;
	}
	.dot {
		width: 7px;
		height: 7px;
		flex: 0 0 auto;
		border-radius: 50%;
		background: var(--dot-color);
		box-shadow: 0 0 10px var(--dot-color);
	}
	.track {
		height: 7px;
		overflow: hidden;
		border-radius: 99px;
		background: var(--elevated);
	}
	.bar {
		display: block;
		height: 100%;
		min-width: 0;
		border-radius: inherit;
	}
	.model-bar {
		background: linear-gradient(90deg, var(--accent), #a78bfa);
	}
	.histogram {
		display: grid;
		grid-template-columns: repeat(5, minmax(42px, 1fr));
		gap: var(--s4);
		align-items: end;
		min-height: 150px;
	}
	.histogram-column {
		display: grid;
		gap: var(--s2);
		justify-items: center;
		color: var(--muted);
		font-size: 11px;
	}
	.histogram-column strong {
		color: var(--text);
		font-size: 12px;
		font-variant-numeric: tabular-nums;
	}
	.histogram-track {
		display: flex;
		align-items: end;
		width: 100%;
		height: 96px;
		border-bottom: 1px solid var(--border-strong);
		background: repeating-linear-gradient(
			to top,
			transparent 0 23px,
			rgba(255, 255, 255, 0.04) 24px
		);
	}
	.histogram-track span {
		display: block;
		width: 100%;
		min-height: 0;
		border-radius: var(--r2) var(--r2) 0 0;
		background: linear-gradient(180deg, var(--accent), #5b6da8);
	}
	.empty {
		padding: var(--s5) 0;
		text-align: center;
	}
	.tables {
		align-items: start;
	}
	.tables section {
		flex: 1 1 350px;
	}
	.tables .wide {
		flex-basis: 100%;
	}
	table {
		width: 100%;
		border-collapse: collapse;
	}
	th,
	td {
		padding: var(--s2);
		border-bottom: 1px solid var(--border);
		text-align: left;
		font-size: 12px;
	}
	th {
		color: var(--muted);
		font-weight: 500;
	}
	@media (max-width: 640px) {
		.charts {
			grid-template-columns: 1fr;
		}
		.wide-chart {
			grid-column: auto;
		}
		.heading {
			align-items: start;
			flex-direction: column;
		}
	}
</style>
