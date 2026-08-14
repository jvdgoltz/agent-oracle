<script lang="ts">
	import { getBehaviorStats, type BehaviorReport, type BehaviorSummary } from '$lib/api';

	/** Signals exposed by OMP's user-metrics classifier. */
	const SIGNALS: { key: keyof BehaviorSummary; rate: keyof BehaviorSummary; label: string }[] = [
		{ key: 'yelling', rate: 'yelling_rate', label: 'Yelling' },
		{ key: 'profanity', rate: 'profanity_rate', label: 'Profanity' },
		{ key: 'anguish', rate: 'anguish_rate', label: 'Anguish' },
		{ key: 'negation', rate: 'negation_rate', label: 'Negation' },
		{ key: 'repetition', rate: 'repetition_rate', label: 'Repetition' },
		{ key: 'blame', rate: 'blame_rate', label: 'Blame' }
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
		<div class="cards">
			{#each SIGNALS as signal (signal.key)}
				<div class="card">
					<span>{signal.label}</span><strong>{rate(report.totals[signal.rate] as number)}</strong
					><small>{report.totals[signal.key] as number} hits</small>
				</div>
			{/each}
		</div>

		<div class="tables">
			<section>
				<h2>By model</h2>
				<p class="table-note">Based on the previous eligible assistant response.</p>
				<div class="table-scroll">
					<table>
						<thead
							><tr
								><th>Model</th><th>Messages</th><th>Yelling</th><th>Profanity</th><th>Anguish</th
								><th>Negation</th><th>Repetition</th><th>Blame</th></tr
							></thead
						><tbody>
							{#each report.models as row (row.model)}<tr
									><td><code>{row.model}</code></td><td>{row.user_messages}</td><td
										>{rate(row.yelling_rate)}</td
									><td>{rate(row.profanity_rate)}</td><td>{rate(row.anguish_rate)}</td><td
										>{rate(row.negation_rate)}</td
									><td>{rate(row.repetition_rate)}</td><td>{rate(row.blame_rate)}</td></tr
								>{/each}
						</tbody>
					</table>
				</div>
			</section>
			<section>
				<h2>By agent</h2>
				<table>
					<thead><tr><th>Agent</th><th>Messages</th><th>Negation</th></tr></thead><tbody>
						{#each report.agents as row (row.agent)}<tr
								><td>{row.agent}</td><td>{row.user_messages}</td><td>{rate(row.negation_rate)}</td
								></tr
							>{/each}
					</tbody>
				</table>
			</section>
			<section>
				<h2>By project</h2>
				<table>
					<thead><tr><th>Project</th><th>Messages</th><th>Negation</th></tr></thead><tbody>
						{#each report.projects as row (row.cwd)}<tr
								><td><code>{row.cwd}</code></td><td>{row.user_messages}</td><td
									>{rate(row.negation_rate)}</td
								></tr
							>{/each}
					</tbody>
				</table>
			</section>
			<section>
				<h2>Daily trend</h2>
				<table>
					<thead><tr><th>Date</th><th>Messages</th><th>Negation</th></tr></thead>
					<tbody>
						{#each report.daily as row (row.date)}
							<tr
								><td>{row.date}</td><td>{row.user_messages}</td><td>{rate(row.negation_rate)}</td
								></tr
							>
						{/each}
					</tbody>
				</table>
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
</style>
