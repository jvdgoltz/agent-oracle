<script lang="ts">
	import { onMount } from 'svelte';
	import { createBrowserState, createSessionBrowser } from '$lib/session-browser';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import AgentBadge from '$lib/AgentBadge.svelte';
	import {
		getSessions,
		getResumableAgentSessions,
		search,
		fetchSearchSummary,
		startAgentSession,
		type SearchMode,
		type SessionSummary
	} from '$lib/api';
	import { relativeTime } from '$lib/format';

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
		{ value: 'omp', label: 'OMP' },
		{ value: 'pi', label: 'Pi' }
	];

	const archive = $state(createBrowserState());
	const browser = createSessionBrowser(archive, { getSessions, search, fetchSearchSummary });

	let agentMessage = $state('');
	let resumeCandidates: SessionSummary[] = $state([]);
	let selectedResumeThreadId = $state('');
	let agentAvailabilityLoading = $state(true);
	let agentSubmitting = $state(false);
	let agentError = $state<string | null>(null);

	/** Load archived Codex threads from this repository that can be resumed. */
	async function loadResumeCandidates() {
		agentAvailabilityLoading = true;
		agentError = null;
		try {
			const data = await getResumableAgentSessions();
			resumeCandidates = data.sessions;
		} catch (e) {
			agentError = e instanceof Error ? e.message : 'Could not load resumable Codex sessions.';
		} finally {
			agentAvailabilityLoading = false;
		}
	}

	/** Start a new Codex thread or continue the selected archived thread. */
	async function startAgent() {
		const message = agentMessage.trim();
		if (!message) {
			agentError = 'Describe what you want Codex to work on.';
			return;
		}
		agentSubmitting = true;
		agentError = null;
		try {
			const session = await startAgentSession(message, selectedResumeThreadId || undefined);
			await goto(resolve(`/agent/${encodeURIComponent(session.thread_id)}`));
		} catch (e) {
			agentError = e instanceof Error ? e.message : 'Could not start the Codex agent.';
		} finally {
			agentSubmitting = false;
		}
	}

	onMount(() => {
		void browser.loadFeed();
		void loadResumeCandidates();
		return browser.cancel;
	});
</script>

{#snippet pagination()}
	<nav class="pagination" aria-label="Session pages">
		<span class="muted"
			>{archive.offset + 1}–{archive.offset + archive.sessions.length} of {archive.total.toLocaleString()}</span
		>
		{#if archive.total > browser.pageSize}
			<button
				type="button"
				aria-label="Previous session page"
				disabled={archive.offset === 0 || archive.loading}
				onclick={() => browser.changePage(-1)}>←</button
			>
			<button
				type="button"
				aria-label="Next session page"
				disabled={archive.offset + browser.pageSize >= archive.total || archive.loading}
				onclick={() => browser.changePage(1)}>→</button
			>
		{/if}
	</nav>
{/snippet}

<div class="home">
	<!-- ── Search bar ─────────────────────────────────────────────── -->
	<form
		class="search-form"
		onsubmit={(e: SubmitEvent) => {
			e.preventDefault();
			browser.runSearch();
		}}
	>
		<div class="search-bar">
			<span class="search-icon" aria-hidden="true">⌕</span>
			<input
				class="search-input"
				type="search"
				aria-label="Search agent sessions"
				oninput={(event) => {
					if (!event.currentTarget.value) void browser.clearSearch();
				}}
				placeholder="Search agent sessions…"
				bind:value={archive.query}
			/>
			{#if archive.query || archive.submittedQuery}
				<button
					class="clear-btn"
					type="button"
					aria-label="Clear search"
					onclick={browser.clearSearch}>✕</button
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
						class="pill {archive.mode === m.value ? 'active' : ''}"
						onclick={() => browser.changeMode(m.value)}
						aria-pressed={archive.mode === m.value}
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
						class="pill {archive.agent === a.value ? 'active' : ''}"
						onclick={() => browser.changeAgent(a.value)}
						aria-pressed={archive.agent === a.value}
					>
						{a.label}
					</button>
				{/each}
			</div>

			{#if archive.query.trim()}
				<button class="search-btn" type="submit">Search</button>
			{/if}
		</div>
	</form>

	<!-- ── Search results ────────────────────────────────────────── -->
	{#if archive.searching}
		<div class="feed-header" role="status"><span class="muted">Searching…</span></div>
	{:else if archive.error}
		<p class="error" role="alert">{archive.error}</p>
	{:else if archive.submittedQuery}
		<div class="feed-header">
			<h2 class="section-title">
				{archive.results.length} result{archive.results.length !== 1 ? 's' : ''} for
				<em>"{archive.submittedQuery}"</em>
			</h2>
		</div>
		{#if archive.results.length === 0}
			<div class="empty-state">
				<p class="empty-icon">⊘</p>
				<p class="empty-title">No results found</p>
				<p class="empty-sub">Try a different query or switch to a different search mode.</p>
			</div>
		{:else}
			{#if archive.summaryLoading}
				<div class="ai-summary ai-summary-loading">
					<div class="ai-summary-icon">✦</div>
					<div class="ai-summary-placeholder">
						<span class="shimmer-line"></span>
						<span class="shimmer-line short"></span>
					</div>
				</div>
			{:else if archive.aiSummary}
				<div class="ai-summary">
					<div class="ai-summary-icon">✦</div>
					<p class="ai-summary-text">{archive.aiSummary}</p>
				</div>
			{/if}
			<ul class="card-list">
				{#each archive.results as result, i (result.session_id + i)}
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
							{#if result.title}
								<h3 class="card-title">{result.title}</h3>
							{/if}
							{#if result.summary}
								<p class="card-summary">{result.summary}</p>
							{/if}
							{#if result.message_snippets.length > 0}
								<ul class="msg-snippets">
									{#each result.message_snippets as snip, i (snip + i)}
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
	{:else if archive.loading}
		<div class="feed-header" role="status"><span class="muted">Loading sessions…</span></div>
	{:else if archive.sessions.length === 0}
		<div class="empty-state">
			<p class="empty-icon">◈</p>
			<p class="empty-title">
				{archive.agent ? 'No sessions for this agent' : 'No sessions indexed yet'}
			</p>
			<p class="empty-sub">
				{archive.agent
					? 'Choose another agent or All to browse the archive.'
					: 'Archived coding sessions will appear here once the watcher imports them.'}
			</p>
		</div>
	{:else}
		<div class="feed-header">
			<h2 class="section-title">Recent sessions</h2>
			{@render pagination()}
		</div>
		<ul class="card-list">
			{#each archive.sessions as session (session.id)}
				<li>
					<a class="card" href={resolve(`/sessions/${session.id}`)}>
						<div class="card-row">
							<AgentBadge agent={session.agent} />
							<code class="cwd">{session.cwd}</code>
							<time class="time ml-auto">{relativeTime(session.started_at)}</time>
						</div>
						{#if session.title}
							<h3 class="card-title">{session.title}</h3>
						{/if}
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
					{#if session.review_sessions.length > 0}
						<div class="review-sessions">
							<span class="muted">Reviews</span>
							{#each session.review_sessions as review (review.id)}
								<a href={resolve(`/sessions/${review.id}`)}>{relativeTime(review.started_at)}</a>
							{/each}
						</div>
					{/if}
				</li>
			{/each}
		</ul>
		{#if archive.total > browser.pageSize}
			<div class="feed-footer">{@render pagination()}</div>
		{/if}
	{/if}

	<!-- ── Codex launcher ─────────────────────────────────────────── -->
	<section class="agent-launcher" aria-labelledby="agent-launcher-title">
		<div class="agent-launcher-heading">
			<div>
				<h2 class="section-title" id="agent-launcher-title">Start a Codex agent</h2>
				<p class="agent-launcher-subtitle">Give Codex a task in this repository.</p>
			</div>
		</div>
		<form
			class="agent-launcher-form"
			onsubmit={(e: SubmitEvent) => {
				e.preventDefault();
				startAgent();
			}}
		>
			<textarea
				class="agent-message"
				aria-label="Task for Codex"
				bind:value={agentMessage}
				placeholder="What should Codex work on?"
				rows="3"
				disabled={agentSubmitting}></textarea>
			<label class="resume-label" for="resume-session"
				>Resume candidates: Codex sessions in this repository.</label
			>
			<select
				class="resume-select"
				id="resume-session"
				bind:value={selectedResumeThreadId}
				disabled={agentAvailabilityLoading || agentSubmitting}
			>
				<option value="">Start a new session</option>
				{#each resumeCandidates as candidate (candidate.id)}
					<option value={candidate.id}>
						Resume {candidate.title || candidate.summary || candidate.id}
					</option>
				{/each}
			</select>
			<div class="agent-launcher-actions">
				{#if agentAvailabilityLoading}
					<span class="muted">Loading resume candidates…</span>
				{:else if !agentError && resumeCandidates.length === 0}
					<span class="muted">No archived Codex sessions are available to resume.</span>
				{/if}
				<button class="agent-start-btn" type="submit" disabled={agentSubmitting}>
					{agentSubmitting ? 'Starting…' : selectedResumeThreadId ? 'Resume Codex' : 'Start Codex'}
				</button>
			</div>
		</form>
		{#if agentError}
			<p class="error agent-error" role="alert">{agentError}</p>
		{/if}
	</section>
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
		min-width: 0;
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
		flex-wrap: wrap;
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
		flex-wrap: wrap;
		justify-content: space-between;
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
		overflow-wrap: anywhere;
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
		max-width: min(380px, 100%);
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
		max-width: 100%;
		overflow-wrap: anywhere;
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

	.card-title {
		margin: 0;
		font-size: 15px;
		font-weight: 600;
		color: var(--text);
		display: -webkit-box;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
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

	/* ── Empty archive ─────────────────────────────────────────────── */
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

	/* ── Codex launcher ─────────────────────────────────────────── */
	.agent-launcher {
		display: flex;
		flex-direction: column;
		gap: var(--s3);
		padding: var(--s4) var(--s5);
		background-color: var(--surface);
		border: 1px solid var(--border-strong);
		border-radius: var(--r3);
	}

	.agent-launcher-heading {
		display: flex;
		align-items: baseline;
	}

	.agent-launcher-subtitle {
		margin: var(--s1) 0 0;
		font-size: 13px;
		color: var(--muted);
	}

	.agent-launcher-form {
		display: flex;
		flex-direction: column;
		gap: var(--s2);
	}

	.agent-message,
	.resume-select {
		box-sizing: border-box;
		width: 100%;
		border: 1px solid var(--border);
		border-radius: var(--r2);
		background-color: var(--elevated);
		color: var(--text);
		font: inherit;
	}

	.agent-message {
		resize: vertical;
		padding: var(--s3);
		line-height: 1.45;
	}

	.agent-message:focus,
	.resume-select:focus {
		outline: none;
		border-color: var(--accent);
		box-shadow: 0 0 0 3px var(--accent-glow);
	}

	.resume-label {
		font-size: 12px;
		color: var(--muted);
	}

	.resume-select {
		padding: var(--s2) var(--s3);
		font-size: 13px;
	}

	.agent-launcher-actions {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--s3);
		min-height: 28px;
	}

	.agent-start-btn {
		margin-left: auto;
		padding: 5px 14px;
		border: 1px solid var(--accent);
		border-radius: var(--r2);
		background-color: var(--accent-dim);
		color: var(--accent);
		font: inherit;
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
	}

	.agent-start-btn:hover:not(:disabled) {
		background-color: var(--accent);
		color: #0d0d0f;
	}

	.agent-start-btn:disabled,
	.agent-message:disabled,
	.resume-select:disabled {
		cursor: not-allowed;
		opacity: 0.65;
	}

	.agent-error {
		margin: 0;
	}

	.pagination {
		display: flex;
		align-items: center;
		gap: var(--s2);
	}
	.pagination button {
		width: 30px;
		height: 30px;
		border: 1px solid var(--border-strong);
		border-radius: var(--r2);
		background: var(--surface);
		color: var(--text);
		cursor: pointer;
	}
	.pagination button:hover:not(:disabled) {
		background: var(--hover);
		border-color: var(--accent);
	}
	.pagination button:disabled {
		opacity: 0.35;
		cursor: default;
	}
	.feed-footer {
		display: flex;
		justify-content: flex-end;
	}
	.review-sessions {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--s3);
		padding: var(--s2) var(--s5);
		background: var(--surface);
		border-top: 1px solid var(--border);
		font-size: 12px;
	}
	@media (max-width: 480px) {
		.card,
		.agent-launcher,
		.ai-summary {
			padding: var(--s3);
		}
		.card-row {
			gap: var(--s2);
		}
		.agent-launcher-actions {
			flex-wrap: wrap;
		}
	}
</style>
