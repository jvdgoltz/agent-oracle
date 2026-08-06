<script lang="ts">
	import { resolve } from '$app/paths';
	import AgentBadge from '$lib/AgentBadge.svelte';
	import {
		getSessions,
		search,
		fetchSearchSummary,
		type SearchMode,
		type SearchResult,
		type SessionSummary
	} from '$lib/api';
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
		{ value: '', label: 'All' },
		{ value: 'codex', label: 'Codex' },
		{ value: 'factory', label: 'Factory' },
		{ value: 'claude', label: 'Claude' },
		{ value: 'omp', label: 'OMP' }
	];

	let sessions: SessionSummary[] = $state([]);
	let loading = $state(true);

	let query = $state('');
	let mode: SearchMode = $state('hybrid');
	let agent = $state('');
	let searching = $state(false);
	let results: SearchResult[] = $state([]);
	let aiSummary = $state('');
	let summaryLoading = $state(false);
	let error = $state<string | null>(null);

	/**
	 * Load the recent session feed, optionally filtered by agent.
	 */
	async function loadFeed() {
		loading = true;
		error = null;
		try {
			const agentParam = agent || undefined;
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
	 * Results are shown immediately; the AI summary is fetched separately.
	 */
	async function runSearch() {
		if (!query.trim()) {
			searching = false;
			results = [];
			aiSummary = '';
			summaryLoading = false;
			return;
		}
		searching = true;
		error = null;
		aiSummary = '';
		summaryLoading = false;
		try {
			const data = await search(query.trim(), mode, 20, agent || undefined);
			results = data.results;
			// Fetch AI summary separately so results render immediately.
			if (results.length > 0) {
				summaryLoading = true;
				fetchSearchSummary(query.trim(), results)
					.then((summary) => {
						aiSummary = summary;
					})
					.catch(() => {
						aiSummary = '';
					})
					.finally(() => {
						summaryLoading = false;
					});
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Search failed';
			results = [];
			aiSummary = '';
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

<div class="home">
	<!-- ── Search bar ─────────────────────────────────────────────── -->
	<form
		class="search-form"
		onsubmit={(e: SubmitEvent) => {
			e.preventDefault();
			runSearch();
		}}
	>
		<div class="search-bar">
			<span class="search-icon" aria-hidden="true">⌕</span>
			<input
				class="search-input"
				type="search"
				placeholder="Search agent sessions…"
				bind:value={query}
				onkeydown={(e) => e.key === 'Enter' && runSearch()}
			/>
			{#if query}
				<button
					class="clear-btn"
					type="button"
					aria-label="Clear search"
					onclick={() => {
						query = '';
						results = [];
					}}>✕</button
				>
			{/if}
		</div>

		<!-- ── Filter pills ──────────────────────────────────────── -->
		<div class="filters">
			<div class="filter-group">
				<span class="filter-label">Mode</span>
				{#each MODES as m (m.value)}
					<button
						type="button"
						class="pill {mode === m.value ? 'active' : ''}"
						onclick={() => (mode = m.value)}
					>
						{m.label}
					</button>
				{/each}
			</div>

			<div class="filter-group">
				<span class="filter-label">Agent</span>
				{#each AGENTS as a (a.value)}
					<button
						type="button"
						class="pill {agent === a.value ? 'active' : ''}"
						onclick={() => (agent = a.value)}
					>
						{a.label}
					</button>
				{/each}
			</div>

			{#if query.trim()}
				<button class="search-btn" type="submit">Search</button>
			{/if}
		</div>
	</form>

	<!-- ── Feedback ──────────────────────────────────────────────── -->
	{#if error}
		<p class="error">{error}</p>
	{/if}

	<!-- ── Search results ────────────────────────────────────────── -->
	{#if searching}
		<div class="feed-header"><span class="muted">Searching…</span></div>
	{:else if query.trim()}
		<div class="feed-header">
			<h2 class="section-title">
				{results.length} result{results.length !== 1 ? 's' : ''} for <em>"{query.trim()}"</em>
			</h2>
		</div>
		{#if results.length === 0}
			<div class="empty-state">
				<p class="empty-icon">⊘</p>
				<p class="empty-title">No results found</p>
				<p class="empty-sub">Try a different query or switch to a different search mode.</p>
			</div>
		{:else}
			{#if summaryLoading}
				<div class="ai-summary ai-summary-loading">
					<div class="ai-summary-icon">✦</div>
					<div class="ai-summary-placeholder">
						<span class="shimmer-line"></span>
						<span class="shimmer-line short"></span>
					</div>
				</div>
			{:else if aiSummary}
				<div class="ai-summary">
					<div class="ai-summary-icon">✦</div>
					<p class="ai-summary-text">{aiSummary}</p>
				</div>
			{/if}
			<ul class="card-list">
				{#each results as result, i (result.session_id + i)}
					<li>
						<a class="card" href={resolve(`/sessions/${result.session_id}`)}>
							<div class="card-row">
								{#if result.agent}
									<AgentBadge agent={result.agent} />
								{/if}
								{#if result.cwd}
									<code class="cwd">{result.cwd}</code>
								{/if}
								<span class="score-badge">score {result.score.toFixed(3)}</span>
								{#if result.started_at}
									<time class="time ml-auto">{relativeTime(result.started_at)}</time>
								{/if}
							</div>
							{#if result.summary}
								<p class="card-summary">{result.summary}</p>
							{/if}
							{#if result.message_snippets.length > 0}
								<ul class="msg-snippets">
									{#each result.message_snippets as snip (snip)}
										<li class="msg-snippet">{snip}</li>
									{/each}
								</ul>
							{:else if result.snippet}
								<p class="snippet">{result.snippet}</p>
							{/if}
							{#if result.entities.length > 0}
								<div class="entities">
									{#each result.entities as entity (entity.type + entity.value)}
										<span class="entity-tag"
											><span class="entity-type">{entity.type}</span>{entity.value}</span
										>
									{/each}
								</div>
							{/if}
						</a>
					</li>
				{/each}
			</ul>
		{/if}

		<!-- ── Recent feed ───────────────────────────────────────────── -->
	{:else if loading}
		<div class="feed-header"><span class="muted">Loading…</span></div>
	{:else if sessions.length === 0}
		<div class="empty-state">
			<p class="empty-icon">◈</p>
			<p class="empty-title">No sessions indexed yet</p>
			<p class="empty-sub">
				Start the watcher with <code>uv run agent-oracle</code> and open a Codex, Factory, or Claude session.
			</p>
		</div>
	{:else}
		<div class="feed-header">
			<h2 class="section-title">Recent sessions</h2>
			<span class="muted">{sessions.length} session{sessions.length !== 1 ? 's' : ''}</span>
		</div>
		<ul class="card-list">
			{#each sessions as session (session.id)}
				<li>
					<a class="card" href={resolve(`/sessions/${session.id}`)}>
						<div class="card-row">
							<AgentBadge agent={session.agent} />
							<code class="cwd">{session.cwd}</code>
							<time class="time ml-auto">{relativeTime(session.started_at)}</time>
						</div>
						{#if session.summary}
							<p class="card-summary">{session.summary}</p>
						{/if}
						{#if session.entities.length > 0}
							<div class="entities">
								{#each session.entities as entity (entity.type + entity.value)}
									<span class="entity-tag"
										><span class="entity-type">{entity.type}</span>{entity.value}</span
									>
								{/each}
							</div>
						{/if}
					</a>
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.home {
		display: flex;
		flex-direction: column;
		gap: var(--s5);
	}

	/* ── Search ──────────────────────────────────────────────────── */
	.search-form {
		display: flex;
		flex-direction: column;
		gap: var(--s3);
	}

	.search-bar {
		display: flex;
		align-items: center;
		gap: var(--s2);
		background-color: var(--surface);
		border: 1px solid var(--border-strong);
		border-radius: var(--r3);
		padding: 0 var(--s3);
		transition: border-color 0.15s;
	}

	.search-bar:focus-within {
		border-color: var(--accent);
		box-shadow: 0 0 0 3px var(--accent-glow);
	}

	.search-icon {
		font-size: 16px;
		color: var(--muted);
		flex-shrink: 0;
		user-select: none;
	}

	.search-input {
		flex: 1;
		background: none;
		border: none;
		outline: none;
		color: var(--text);
		font-family: var(--font-sans);
		font-size: 14px;
		padding: 10px 0;
	}

	.search-input::placeholder {
		color: var(--muted);
	}

	/* Remove native clear button on search inputs */
	.search-input::-webkit-search-cancel-button {
		display: none;
	}

	.clear-btn {
		background: none;
		border: none;
		color: var(--muted);
		font-size: 12px;
		cursor: pointer;
		padding: var(--s1);
		line-height: 1;
		border-radius: var(--r1);
		transition: color 0.15s;
	}

	.clear-btn:hover {
		color: var(--text);
	}

	/* ── Filter pills ────────────────────────────────────────────── */
	.filters {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--s3);
	}

	.filter-group {
		display: flex;
		align-items: center;
		gap: var(--s1);
	}

	.filter-label {
		font-size: 11px;
		font-weight: 500;
		color: var(--muted);
		letter-spacing: 0.05em;
		text-transform: uppercase;
		margin-right: var(--s1);
		white-space: nowrap;
	}

	.pill {
		display: inline-flex;
		align-items: center;
		padding: 3px 10px;
		border-radius: 999px;
		border: 1px solid var(--border);
		background-color: transparent;
		color: var(--muted);
		font-family: var(--font-sans);
		font-size: 12px;
		font-weight: 500;
		cursor: pointer;
		transition:
			border-color 0.15s,
			color 0.15s,
			background-color 0.15s;
		white-space: nowrap;
	}

	.pill:hover {
		border-color: var(--border-strong);
		color: var(--text);
	}

	.pill.active {
		border-color: var(--accent);
		background-color: var(--accent-dim);
		color: var(--accent);
	}

	.search-btn {
		margin-left: auto;
		padding: 4px 14px;
		border-radius: var(--r2);
		border: 1px solid var(--accent);
		background-color: var(--accent-dim);
		color: var(--accent);
		font-family: var(--font-sans);
		font-size: 13px;
		font-weight: 500;
		cursor: pointer;
		transition:
			background-color 0.15s,
			color 0.15s;
	}

	.search-btn:hover {
		background-color: var(--accent);
		color: #0d0d0f;
	}

	/* ── Feed header ─────────────────────────────────────────────── */
	.feed-header {
		display: flex;
		align-items: baseline;
		gap: var(--s3);
	}

	.section-title {
		margin: 0;
		font-size: 13px;
		font-weight: 600;
		color: var(--text);
	}

	.section-title em {
		font-style: normal;
		color: var(--accent);
	}

	.muted {
		font-size: 12px;
		color: var(--muted);
	}

	/* ── Cards ───────────────────────────────────────────────────── */
	.card-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 1px;
		border: 1px solid var(--border);
		border-radius: var(--r3);
		overflow: hidden;
	}

	.card {
		display: flex;
		flex-direction: column;
		gap: var(--s2);
		padding: var(--s4) var(--s5);
		background-color: var(--surface);
		color: var(--text);
		text-decoration: none;
		transition: background-color 0.1s;
		border-bottom: 1px solid var(--border);
	}

	.card-list li:last-child .card {
		border-bottom: none;
	}

	.card:hover {
		background-color: var(--hover);
		text-decoration: none;
	}

	.card-row {
		display: flex;
		align-items: center;
		gap: var(--s3);
		flex-wrap: wrap;
	}

	.cwd {
		font-family: var(--font-mono);
		font-size: 12px;
		color: var(--muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 380px;
	}

	.ml-auto {
		margin-left: auto;
	}

	.entities {
		display: flex;
		flex-wrap: wrap;
		gap: var(--s1);
	}

	.entity-tag {
		display: inline-flex;
		align-items: center;
		gap: var(--s1);
		padding: 1px 8px;
		border-radius: 999px;
		border: 1px solid var(--border);
		background-color: var(--elevated);
		font-size: 11px;
		color: var(--muted);
	}

	.entity-type {
		font-size: 9px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--subtle);
	}

	.card-summary {
		margin: 0;
		font-size: 13px;
		color: var(--muted);
		display: -webkit-box;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}

	/* ── Search result cards ─────────────────────────────────────── */

	.ai-summary {
		display: flex;
		gap: var(--s3);
		padding: var(--s4) var(--s5);
		background-color: var(--surface);
		border: 1px solid var(--accent);
		border-radius: var(--r3);
		margin-bottom: var(--s3);
	}

	.ai-summary-icon {
		flex-shrink: 0;
		font-size: 16px;
		color: var(--accent);
		line-height: 1.4;
	}

	.ai-summary-text {
		margin: 0;
		font-size: 13px;
		line-height: 1.5;
		color: var(--text);
	}

	.ai-summary-loading {
		border-color: var(--border);
	}

	.ai-summary-placeholder {
		display: flex;
		flex-direction: column;
		gap: var(--s2);
		flex: 1;
	}

	.shimmer-line {
		height: 12px;
		border-radius: var(--r1);
		background: linear-gradient(
			90deg,
			var(--border) 0%,
			var(--border-strong) 50%,
			var(--border) 100%
		);
		background-size: 200% 100%;
		animation: shimmer 1.5s infinite;
	}

	.shimmer-line.short {
		width: 60%;
	}

	@keyframes shimmer {
		0% {
			background-position: 200% 0;
		}
		100% {
			background-position: -200% 0;
		}
	}

	.score-badge {
		margin-left: auto;
		font-size: 11px;
		font-family: var(--font-mono);
		color: var(--muted);
		background-color: var(--elevated);
		border: 1px solid var(--border);
		border-radius: var(--r1);
		padding: 1px 6px;
	}

	.snippet {
		margin: 0;
		font-size: 13px;
		color: var(--muted);
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}

	.msg-snippets {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--s1);
	}

	.msg-snippet {
		font-size: 12px;
		color: var(--muted);
		padding-left: var(--s3);
		border-left: 2px solid var(--border);
		display: -webkit-box;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
		line-height: 1.4;
	}

	/* ── Empty state ─────────────────────────────────────────────── */
	.empty-state {
		padding: var(--s8) var(--s5);
		text-align: center;
		border: 1px dashed var(--border-strong);
		border-radius: var(--r3);
	}

	.empty-icon {
		margin: 0 0 var(--s3);
		font-size: 32px;
		color: var(--subtle);
		line-height: 1;
	}

	.empty-title {
		margin: 0 0 var(--s2);
		font-size: 15px;
		font-weight: 600;
		color: var(--text);
	}

	.empty-sub {
		margin: 0;
		font-size: 13px;
		color: var(--muted);
		max-width: 420px;
		margin-inline: auto;
	}
</style>
