<script lang="ts">
	import { page } from '$app/state';
	import AgentBadge from '$lib/AgentBadge.svelte';
	import { getSession, type Message, type SessionDetail } from '$lib/api';
	import { relativeTime } from '$lib/format';
	import { marked } from 'marked';

	/** The `id` route parameter for this session. */
	const id: string = page.params.id ?? '';

	let session: SessionDetail | null = $state(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	/** Messages ordered newest-first for display. */
	let orderedMessages: { entry: Message; key: string; html: string }[] = $state([]);

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
				key: `${entry.timestamp}-${entry.content.length}`,
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

	/**
	 * Map a raw role to a display label; user roles get "You", everything
	 * else is treated as an assistant-style message.
	 * @param role - Raw role string from the backend.
	 * @returns A display label for the message header.
	 */
	function roleLabel(role: string): string {
		return role === 'user' ? 'You' : 'Assistant';
	}

	/**
	 * Determine bubble alignment from the raw role.
	 * @param role - Raw role string from the backend.
	 * @returns True when the message is user-authored (rendered on the right).
	 */
	function isUser(role: string): boolean {
		return role === 'user';
	}
</script>

<main class="page">
	{#if loading}
		<p class="status">Loading session…</p>
	{:else if error}
		<p class="error">{error}</p>
	{:else if session}
		<header class="header">
			<div class="row">
				<AgentBadge agent={session.agent} />
				<time class="time">{relativeTime(session.started_at)}</time>
			</div>
			<span class="cwd">{session.cwd}</span>
			{#if session.summary}
				<p class="summary">{session.summary}</p>
			{/if}
			{#if session.entities.length > 0}
				<div class="entities">
					{#each session.entities as entity (entity.value)}
						<span class="tag">{entity.type}: {entity.value}</span>
					{/each}
				</div>
			{/if}
		</header>

		<ul class="messages">
			{#each orderedMessages as { entry, key, html } (key)}
				<li class="message {isUser(entry.role) ? 'user' : 'assistant'}">
					<div class="bubble">
						<div class="meta">
							<span class="role">{roleLabel(entry.role)}</span>
							<time class="time">{relativeTime(entry.timestamp)}</time>
						</div>
						<div class="content">
							<!-- eslint-disable-next-line svelte/no-at-html-tags -->
							{@html html}
						</div>
					</div>
				</li>
			{/each}
		</ul>
	{:else}
		<p class="status">Session not found.</p>
	{/if}
</main>

<style>
	.page {
		max-width: 720px;
		margin: 0 auto;
		padding: 1.5rem;
	}

	.header {
		margin-bottom: 1.5rem;
		padding-bottom: 1rem;
		border-bottom: 1px solid #2e2e2e;
	}

	.summary {
		font-size: 0.875rem;
	}

	.entities {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		margin-top: 0.75rem;
	}

	.tag {
		padding: 0.15em 0.6em;
		border-radius: 999px;
		background-color: #2a2a2a;
		border: 1px solid #3a3a3a;
		font-size: 0.75rem;
	}

	.messages {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 1rem;
	}

	.message {
		display: flex;
	}

	.message.user {
		justify-content: flex-end;
	}

	.bubble {
		max-width: 78%;
		padding: 0.75rem 1rem;
		border-radius: 12px;
		background-color: #242424;
	}

	.message.user .bubble {
		background-color: #2b3a5a;
	}

	.meta {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.4rem;
	}

	.role {
		font-size: 0.75rem;
		font-weight: 600;
	}

	.content {
		font-size: 0.9rem;
		overflow-wrap: anywhere;
	}

	.content :global(p) {
		margin: 0 0 0.75rem;
	}

	.content :global(p:last-child) {
		margin-bottom: 0;
	}
</style>
