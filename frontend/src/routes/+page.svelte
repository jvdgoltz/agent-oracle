<script lang="ts">
	import { resolve } from '$app/paths';
	import AgentBadge from '$lib/AgentBadge.svelte';
	import { getSessions, search, type SearchMode, type SessionSummary } from '$lib/api';
	import { relativeTime } from '$lib/format';

	/** Max sessions loaded for the recent feed. */
	const FEED_LIMIT = 50;

	/** Ordered list of search modes offered by the backend. */
	const MODES: { value: SearchMode; label: string }[] = [
		{ value: 'hybrid', label: 'Hybrid' },
		{ value: 'text', label: 'Text' },
		{ value: 'vector', label: 'Vector' }
	];

	/** Agent filter options; empty string means "all agents". */
	const AGENTS: { value: string; label: string }[] = [
		{ value: '', label: 'All agents' },
		{ value: 'codex', label: 'Codex' },
		{ value: 'factory', label: 'Factory' },
		{ value: 'claude', label: 'Claude' }
	];

	let sessions: SessionSummary[] = $state([]);
	let loading = $state(true);

	let query = $state('');
	let mode: SearchMode = $state('hybrid');
	let agent = $state('');
	let searching = $state(false);
	let results: { session_id: string; snippet: string; score: number }[] = $state([]);
	let error = $state<string | null>(null);

	/**
	 * Load the recent session feed, optionally filtered by agent.
	 */
	async function loadFeed() {
		loading = true;
		error = null;
		try {
			const agentParam = agent || undefined;
			// The feed endpoint has no agent filter today, so filter client-side.
			const data = await getSessions(FEED_LIMIT);
			sessions = agentParam
				? data.sessions.filter((s) => s.agent.toLowerCase().includes(agentParam))
				: data.sessions;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load sessions';
		} finally {
			loading = false;
		}
	}

	/**
	 * Trigger a search across sessions using the current query and mode.
	 */
	async function runSearch() {
		if (!query.trim()) {
			searching = false;
			results = [];
			return;
		}
		searching = true;
		error = null;
		try {
			const data = await search(query.trim(), mode, 20, agent || undefined);
			results = data.results;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Search failed';
			results = [];
		} finally {
			searching = false;
		}
	}

	$effect(() => {
		loadFeed();
	});

	// Reload the feed whenever the agent filter changes while not searching.
	$effect(() => {
		if (!searching) {
			loadFeed();
		}
	});
</script>

<main class="page">
	<h1 class="title">Agent Oracle</h1>

	<div class="controls">
		<form
			class="search"
			onsubmit={(e: SubmitEvent) => {
				e.preventDefault();
				runSearch();
			}}
		>
			<input type="search" placeholder="Search sessions…" bind:value={query} />
			<select aria-label="Search mode" bind:value={mode}>
				{#each MODES as item (item.value)}
					<option value={item.value}>{item.label}</option>
				{/each}
			</select>
			<select aria-label="Agent filter" bind:value={agent}>
				{#each AGENTS as item (item.value)}
					<option value={item.value}>{item.label}</option>
				{/each}
			</select>
			<button type="submit">Search</button>
		</form>
	</div>

	{#if error}
		<p class="error">{error}</p>
	{/if}

	{#if searching}
		<p class="status">Searching…</p>
	{:else if query.trim()}
		<section class="results">
			{#if results.length === 0}
				<p class="status">No results.</p>
			{:else}
				<h2>Search results</h2>
				<ul class="cards">
					{#each results as result (result.session_id)}
						<li>
							<a class="card" href={resolve(`/sessions/${result.session_id}`)}>
								<span class="meta">
									Score {result.score.toFixed(3)}
								</span>
								<span class="snippet">{result.snippet}</span>
							</a>
						</li>
					{/each}
				</ul>
			{/if}
		</section>
	{:else if loading}
		<p class="status">Loading sessions…</p>
	{:else if sessions.length === 0}
		<p class="status">No sessions yet.</p>
	{:else}
		<section>
			<h2>Recent sessions</h2>
			<ul class="cards">
				{#each sessions as session (session.id)}
					<li>
						<a class="card" href={resolve(`/sessions/${session.id}`)}>
							<span class="row">
								<AgentBadge agent={session.agent} />
								<span class="time">{relativeTime(session.started_at)}</span>
							</span>
							<span class="cwd">{session.cwd}</span>
							<span class="summary">{session.summary}</span>
						</a>
					</li>
				{/each}
			</ul>
		</section>
	{/if}
</main>

<style>
	.page {
		max-width: 720px;
		margin: 0 auto;
		padding: 1.5rem;
	}

	.title {
		margin: 0 0 1rem;
		font-size: 1.5rem;
	}

	.controls {
		margin-bottom: 1.5rem;
	}

	.search {
		display: flex;
		gap: 0.5rem;
	}

	.search input[type='search'] {
		flex: 1;
	}

	.search input,
	.search select,
	.search button {
		padding: 0.4rem 0.6rem;
		border: 1px solid #3a3a3a;
		border-radius: 6px;
		background-color: #242424;
		color: #e6e6e6;
		font-size: 0.875rem;
	}

	.cards {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 0.75rem;
	}

	.card {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		padding: 0.75rem;
		border: 1px solid #2e2e2e;
		border-radius: 8px;
		background-color: #202020;
		color: inherit;
	}

	.card:hover {
		border-color: #7aa2f7;
		text-decoration: none;
	}

	.summary,
	.snippet {
		font-size: 0.875rem;
	}

	.meta {
		font-size: 0.75rem;
		color: #8a8a8a;
	}
</style>
