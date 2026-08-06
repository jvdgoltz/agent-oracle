<script lang="ts">
	import { page } from '$app/state';
	import AgentBadge from '$lib/AgentBadge.svelte';
	import { getSession, type Message, type SessionDetail } from '$lib/api';
	import { relativeTime } from '$lib/format';
	import { marked } from 'marked';
	import { SvelteSet } from 'svelte/reactivity';

	/** The `id` route parameter for this session. */
	const id: string = page.params.id ?? '';

	let session: SessionDetail | null = $state(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	/** Messages ordered newest-first for display. */
	let orderedMessages: { entry: Message; key: string; html: string }[] = $state([]);

	/** Whether non-searchable messages (thinking/system/injected) are shown. */
	let showHidden = $state(false);

	/** Whether the raw session JSON is expanded in the header. */
	let showSessionMeta = $state(false);

	/** Track which messages have their JSON metadata expanded. */
	let expandedMetadata = new SvelteSet<number>();

	/**
	 * Toggle the JSON metadata panel for a message.
	 * @param msgId - The message id.
	 */
	function toggleMetadata(msgId: number): void {
		if (expandedMetadata.has(msgId)) {
			expandedMetadata.delete(msgId);
		} else {
			expandedMetadata.add(msgId);
		}
	}

	/** Count of hidden messages that are not currently shown. */
	let hiddenCount = $derived(
		orderedMessages.filter(
			({ entry }) => entry.is_thinking || entry.is_system_instruction || entry.is_injected
		).length
	);

	/** Visible messages after applying the showHidden filter. */
	let visibleMessages = $derived(
		showHidden
			? orderedMessages
			: orderedMessages.filter(
					({ entry }) => !entry.is_thinking && !entry.is_system_instruction && !entry.is_injected
				)
	);

	/**
	 * Render message content as HTML via marked.
	 * @param content - Raw markdown content.
	 * @returns HTML string for `{@html}` injection.
	 */
	function renderMarkdown(content: string): string {
		return marked.parse(content, { async: false }) as string;
	}

	/**
	 * Load the full session detail once the id parameter is known.
	 */
	async function load() {
		loading = true;
		error = null;
		try {
			if (!id) {
				session = null;
				return;
			}
			session = await getSession(id);
			orderedMessages = [...session.messages].reverse().map((entry) => ({
				entry,
				key: `${entry.id}`,
				html: renderMarkdown(entry.content)
			}));
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load session';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		load();
	});

	/** Human-readable label for a message type. */
	function roleLabel(entry: Message): string {
		if (entry.is_thinking) return 'Thinking';
		if (entry.is_system_instruction) return 'System';
		if (entry.is_injected) return 'Injected';
		return entry.role === 'user' ? 'You' : 'Assistant';
	}

	/** True when the message is user-authored (rendered on the right). */
	function isUser(entry: Message): boolean {
		return entry.role === 'user' && !entry.is_injected;
	}

	/** CSS modifier class for a message bubble. */
	function bubbleClass(entry: Message): string {
		if (entry.is_thinking) return 'thinking';
		if (entry.is_system_instruction) return 'system';
		if (entry.is_injected) return 'injected';
		return isUser(entry) ? 'user' : 'assistant';
	}
</script>

{#if loading}
	<p class="status">Loading session…</p>
{:else if error}
	<p class="error">{error}</p>
{:else if !session}
	<p class="status">Session not found.</p>
{:else}
	<!-- ── Session header ─────────────────────────────────────────── -->
	<header class="session-header">
		<div class="header-top">
			<AgentBadge agent={session.agent} />
			<code class="cwd">{session.cwd}</code>
			<time class="time ml-auto">{relativeTime(session.started_at)}</time>
		</div>

		{#if session.summary}
			<p class="session-summary">{session.summary}</p>
		{/if}

		{#if session.entities && session.entities.length > 0}
			<div class="entities">
				{#each session.entities as entity (entity.type + entity.value)}
					<span class="entity-tag"
						><span class="entity-type">{entity.type}</span>{entity.value}</span
					>
				{/each}
			</div>
		{/if}

		<button class="metadata-toggle" onclick={() => (showSessionMeta = !showSessionMeta)}>
			{showSessionMeta ? '− metadata' : '+ metadata'}
		</button>
		{#if showSessionMeta}
			<pre class="metadata-json">{JSON.stringify(
					session,
					(key, val) => (key === 'messages' ? undefined : val),
					2
				)}</pre>
		{/if}
	</header>

	<!-- ── Hidden messages toggle ────────────────────────────────── -->
	{#if hiddenCount > 0}
		<div class="hidden-bar">
			<span class="hidden-info">
				{hiddenCount} hidden message{hiddenCount !== 1 ? 's' : ''} (thinking / system / injected)
			</span>
			<button class="toggle-btn" onclick={() => (showHidden = !showHidden)}>
				{showHidden ? 'Hide' : 'Show'}
			</button>
		</div>
	{/if}

	<!-- ── Messages ──────────────────────────────────────────────── -->
	{#if visibleMessages.length === 0}
		<p class="status">No messages.</p>
	{:else}
		<ul class="messages">
			{#each visibleMessages as { entry, key, html } (key)}
				<li class="message {bubbleClass(entry)}">
					<div class="bubble">
						<div class="bubble-meta">
							<span class="role-label">{roleLabel(entry)}</span>
							{#if entry.model}
								<span class="model-badge">{entry.model}</span>
							{/if}
							<time class="time ml-auto">{relativeTime(entry.timestamp)}</time>
						</div>
						<div class="content">
							<!-- eslint-disable-next-line svelte/no-at-html-tags -->
							{@html html}
						</div>
						<button class="metadata-toggle" onclick={() => toggleMetadata(entry.id)}>
							{expandedMetadata.has(entry.id) ? '− metadata' : '+ metadata'}
						</button>
						{#if expandedMetadata.has(entry.id)}
							<pre class="metadata-json">{JSON.stringify(entry, null, 2)}</pre>
						{/if}
					</div>
				</li>
			{/each}
		</ul>
	{/if}
{/if}

<style>
	/* ── Session header ──────────────────────────────────────────── */
	.session-header {
		display: flex;
		flex-direction: column;
		gap: var(--s3);
		padding-bottom: var(--s5);
		margin-bottom: var(--s5);
		border-bottom: 1px solid var(--border);
	}

	.header-top {
		display: flex;
		align-items: center;
		gap: var(--s3);
		flex-wrap: wrap;
	}

	.ml-auto {
		margin-left: auto;
	}

	.session-summary {
		margin: 0;
		font-size: 14px;
		color: var(--text);
		line-height: 1.5;
	}

	.entities {
		display: flex;
		flex-wrap: wrap;
		gap: var(--s2);
	}

	.entity-tag {
		display: inline-flex;
		align-items: center;
		gap: var(--s1);
		padding: 2px 8px;
		border-radius: 999px;
		border: 1px solid var(--border);
		background-color: var(--elevated);
		font-size: 12px;
		color: var(--muted);
	}

	.entity-type {
		font-size: 10px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--subtle);
	}

	/* ── Hidden messages bar ─────────────────────────────────────── */
	.hidden-bar {
		display: flex;
		align-items: center;
		gap: var(--s3);
		padding: var(--s2) var(--s4);
		background-color: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--r2);
		margin-bottom: var(--s4);
	}

	.hidden-info {
		font-size: 12px;
		color: var(--muted);
	}

	.toggle-btn {
		margin-left: auto;
		background: none;
		border: 1px solid var(--border-strong);
		border-radius: var(--r2);
		color: var(--muted);
		font-family: var(--font-sans);
		font-size: 12px;
		font-weight: 500;
		padding: 2px 10px;
		cursor: pointer;
		transition:
			border-color 0.15s,
			color 0.15s;
	}

	.toggle-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	/* ── Messages ────────────────────────────────────────────────── */
	.messages {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--s3);
	}

	.message {
		display: flex;
	}

	.message.user {
		justify-content: flex-end;
	}

	/* ── Bubble base ─────────────────────────────────────────────── */
	.bubble {
		max-width: 80%;
		padding: var(--s3) var(--s4);
		border-radius: var(--r4);
		border: 1px solid var(--border);
		background-color: var(--surface);
		display: flex;
		flex-direction: column;
		gap: var(--s2);
	}

	/* User bubble: right-aligned, accent tinted */
	.message.user .bubble {
		background-color: var(--elevated);
		border-color: var(--border-strong);
		border-bottom-right-radius: var(--r1);
	}

	/* Assistant bubble: left-aligned with accent left border */
	.message.assistant .bubble {
		border-left: 2px solid var(--accent);
		border-bottom-left-radius: var(--r1);
		padding-left: calc(var(--s4) - 1px);
	}

	/* Thinking: dim blue tint */
	.message.thinking .bubble {
		background-color: #0d0d1a;
		border-color: #252545;
		border-left: 2px solid #4a4a8a;
		opacity: 0.75;
	}

	/* System: purple tint */
	.message.system .bubble {
		background-color: #120d1a;
		border-color: #2a1f40;
		border-left: 2px solid #6a4fa0;
		opacity: 0.8;
	}

	/* Injected: amber tint */
	.message.injected .bubble {
		background-color: #14120a;
		border-color: #302810;
		border-left: 2px solid #806030;
		opacity: 0.8;
	}

	/* ── Bubble meta row ─────────────────────────────────────────── */
	.bubble-meta {
		display: flex;
		align-items: center;
		gap: var(--s2);
	}

	.role-label {
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--muted);
	}

	.message.user .role-label {
		color: var(--accent);
	}

	.message.assistant .role-label {
		color: var(--accent);
	}

	.model-badge {
		font-family: var(--font-mono);
		font-size: 10px;
		padding: 1px 6px;
		border-radius: var(--r1);
		background-color: var(--elevated);
		border: 1px solid var(--border);
		color: var(--muted);
	}

	/* ── Bubble content ──────────────────────────────────────────── */
	.content {
		font-size: 14px;
		line-height: 1.65;
		overflow-wrap: anywhere;
	}

	.message.thinking .content {
		font-style: italic;
		font-size: 13px;
		color: #8080b0;
	}

	.message.system .content,
	.message.injected .content {
		font-size: 12px;
		color: var(--muted);
		font-family: var(--font-mono);
	}

	/* Markdown resets inside bubbles */
	.content :global(p) {
		margin: 0 0 var(--s3);
	}

	.content :global(p:last-child) {
		margin-bottom: 0;
	}

	.content :global(ul),
	.content :global(ol) {
		margin: 0 0 var(--s3);
		padding-left: var(--s5);
	}

	.content :global(li) {
		margin-bottom: var(--s1);
	}

	.content :global(h1),
	.content :global(h2),
	.content :global(h3) {
		margin: var(--s4) 0 var(--s2);
		font-size: 14px;
		font-weight: 600;
		color: var(--text);
	}

	.content :global(blockquote) {
		border-left: 2px solid var(--border-strong);
		margin: 0 0 var(--s3);
		padding-left: var(--s3);
		color: var(--muted);
	}

	/* ── Metadata toggle ─────────────────────────────────────────── */
	.metadata-toggle {
		align-self: flex-start;
		background: none;
		border: none;
		color: var(--subtle);
		font-family: var(--font-mono);
		font-size: 11px;
		cursor: pointer;
		padding: 0;
		transition: color 0.15s;
	}

	.metadata-toggle:hover {
		color: var(--muted);
	}

	.metadata-json {
		margin: 0;
		padding: var(--s3);
		background-color: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--r2);
		font-size: 11px;
		line-height: 1.5;
		overflow-x: auto;
		color: var(--muted);
	}
</style>
