<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { renderMarkdown } from '$lib/markdown';
	import {
		agentEventsUrl,
		deleteAgentSession,
		getArchivedAgentSession,
		sendAgentMessage,
		stopAgentSession,
		type AgentEvent,
		type Message
	} from '$lib/api';

	/** A visible conversation message or collapsed Codex activity entry. */
	type TranscriptEntry = {
		id: number;
		kind: 'user' | 'assistant' | 'activity' | 'error' | 'notice' | 'turn-notice';
		label: string;
		text: string;
		summary?: string;
		events?: AgentEvent[];
		activityKey?: string;
		isReasoning?: boolean;
	};

	/** Named stream events supported by the backend's stable agent contract. */
	const EVENT_TYPES = [
		'assistant',
		'reasoning',
		'status',
		'item',
		'command',
		'file',
		'usage',
		'retry',
		'error',
		'auth',
		'user',
		'completed'
	] as const;

	const threadId = page.params.id ?? '';
	let entries: TranscriptEntry[] = $state([]);
	let message = $state('');
	let running = $state(false);
	let loading = $state(true);
	let stopping = $state(false);
	let clearing = $state(false);
	let submitting = $state(false);
	let error = $state<string | null>(null);
	let source: EventSource | null = null;
	let nextEntryId = 0;
	let activeAssistantId: number | null = null;
	let receivedAssistant = false;

	/** Display a stored user, assistant, and reasoning transcript without starting Codex. */
	function appendArchivedMessages(messages: Message[]): void {
		for (const entry of messages) {
			if (entry.is_thinking) {
				addEntry('activity', 'Reasoning', entry.content, undefined, entry.content, true);
			} else if (entry.role === 'user' && !entry.is_injected) {
				addEntry('user', 'You', entry.content);
			} else if (entry.role === 'assistant' && !entry.is_system_instruction && !entry.is_injected) {
				addEntry('assistant', 'Codex', entry.content);
			}
		}
	}

	/** Load an archived thread when direct navigation follows a backend restart. */
	async function initialize(): Promise<void> {
		try {
			const session = await getArchivedAgentSession(threadId);
			appendArchivedMessages(session.messages);
			running = false;
		} catch {
			running = true;
			connect();
		} finally {
			loading = false;
		}
	}

	/** Open the SSE stream for the current agent turn. */
	function connect(): void {
		source?.close();
		source = new EventSource(agentEventsUrl(threadId));
		for (const type of EVENT_TYPES) {
			source.addEventListener(type, (event) => receiveEvent(event as MessageEvent<string>));
		}
		source.onerror = (event) => {
			if (event instanceof MessageEvent && event.data) receiveEvent(event as MessageEvent<string>);
		};
	}

	/** Decode and display one structured server-sent event. */
	function receiveEvent(event: MessageEvent<string>): void {
		let agentEvent: AgentEvent;
		try {
			agentEvent = JSON.parse(event.data) as AgentEvent;
		} catch {
			return;
		}
		if (agentEvent.type === 'assistant') {
			appendAssistant(extractText(agentEvent.data), agentEvent.method === 'item/completed');
			return;
		}
		if (agentEvent.type === 'user') {
			addEntry('user', 'You', extractText(agentEvent.data));
			return;
		}
		if (agentEvent.type === 'completed') {
			running = false;
			source?.close();
			source = null;
			if (!receivedAssistant) {
				addEntry(
					'turn-notice',
					'No assistant response',
					'Codex completed this turn without sending a final response.',
					[agentEvent]
				);
			}
			appendActivity('activity', agentEvent);
			return;
		}
		if (agentEvent.type === 'auth' && authMode(agentEvent.data) === 'api_key_fallback') {
			appendActivity('notice', agentEvent);
			return;
		}
		if (agentEvent.type === 'error') {
			const message = extractText(agentEvent.data) || 'The agent turn failed.';
			error = message;
			running = false;
			source?.close();
			source = null;
			addEntry('error', 'Agent error', message, [agentEvent]);
			return;
		}
		appendActivity('activity', agentEvent);
	}

	/** Add a regular transcript entry. */
	function addEntry(
		kind: TranscriptEntry['kind'],
		label: string,
		text: string,
		events?: AgentEvent[],
		summary?: string,
		isReasoning = false
	): void {
		entries = [...entries, { id: nextEntryId++, kind, label, text, summary, events, isReasoning }];
	}

	/** Aggregate stream deltas into one expandable entry for each SDK item. */
	function appendActivity(kind: 'activity' | 'notice', event: AgentEvent): void {
		const activityKey = `${event.type}:${eventIdentifier(event.data) ?? event.method ?? 'stream'}`;
		const text = extractText(event.data);
		const summary = reasoningSummary(event, text);
		const entry = entries.find((candidate) => candidate.activityKey === activityKey);
		if (entry) {
			entries = entries.map((candidate) =>
				candidate.id === entry.id
					? {
							...candidate,
							text: appendText(candidate.text, text),
							summary: appendText(candidate.summary ?? '', summary),
							events: [...(candidate.events ?? []), event]
						}
					: candidate
			);
			return;
		}
		entries = [
			...entries,
			{
				id: nextEntryId++,
				kind,
				label: eventLabel(event),
				text,
				summary,
				events: [event],
				activityKey,
				isReasoning: event.type === 'reasoning'
			}
		];
	}

	/** Append streaming text to the assistant bubble for this turn. */
	function appendAssistant(text: string, isFinal = false): void {
		if (!text) return;
		receivedAssistant = true;
		const entry = entries.find((candidate) => candidate.id === activeAssistantId);
		if (entry) {
			if (isFinal && (entry.text === text || entry.text.endsWith(text))) return;
			entries = entries.map((candidate) =>
				candidate.id === entry.id
					? {
							...candidate,
							text: isFinal && text.startsWith(candidate.text) ? text : candidate.text + text
						}
					: candidate
			);
			return;
		}
		const id = nextEntryId++;
		activeAssistantId = id;
		entries = [...entries, { id, kind: 'assistant', label: 'Codex', text }];
	}

	/** Send the next conversational turn and begin its event stream. */
	async function sendMessage(): Promise<void> {
		const text = message.trim();
		if (!text || running || submitting) return;
		submitting = true;
		error = null;
		try {
			await sendAgentMessage(threadId, text);
			message = '';
			activeAssistantId = null;
			receivedAssistant = false;
			running = true;
			connect();
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Failed to send message';
		} finally {
			submitting = false;
		}
	}

	/** Interrupt the active turn and keep the completed transcript visible. */
	async function stop(): Promise<void> {
		if (!running || stopping) return;
		stopping = true;
		try {
			await stopAgentSession(threadId);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Failed to stop agent';
		} finally {
			stopping = false;
		}
	}

	/** Discard this backend session and return home after it is released. */
	async function startNewSession(): Promise<void> {
		if (clearing || submitting) return;
		clearing = true;
		error = null;
		try {
			await deleteAgentSession(threadId);
			source?.close();
			source = null;
			running = false;
			await goto(resolve('/'));
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Failed to start a new session';
		} finally {
			clearing = false;
		}
	}

	/** Return concise visible text from a structured Codex event payload. */
	function extractText(data: unknown): string {
		if (typeof data === 'string') return data;
		if (!data || typeof data !== 'object') return data === undefined ? '' : String(data);
		const record = data as Record<string, unknown>;
		for (const key of ['delta', 'text', 'content', 'message', 'summary', 'output']) {
			if (typeof record[key] === 'string') return record[key];
		}
		const item = record.item;
		if (item && typeof item === 'object' && typeof (item as { text?: unknown }).text === 'string') {
			return (item as { text: string }).text;
		}
		return '';
	}

	/** Return summary deltas that remain visible while raw reasoning stays collapsed. */
	function reasoningSummary(event: AgentEvent, text: string): string {
		return event.method === 'item/reasoning/summaryTextDelta' ? text : '';
	}

	/** Extract the SDK item or turn identifier used to group stream updates. */
	function eventIdentifier(data: unknown): string | undefined {
		if (!data || typeof data !== 'object') return undefined;
		const record = data as Record<string, unknown>;
		for (const key of ['itemId', 'item_id', 'turnId', 'turn_id', 'id']) {
			if (typeof record[key] === 'string') return record[key];
		}
		const item = record.item;
		return item && typeof item === 'object' && typeof (item as { id?: unknown }).id === 'string'
			? (item as { id: string }).id
			: undefined;
	}

	/** Join stream text without adding blank lines for metadata-only events. */
	function appendText(previous: string, next: string): string {
		if (!next) return previous;
		return previous ? `${previous}\n${next}` : next;
	}

	/** Return the authentication mode from an auth event payload. */
	function authMode(data: unknown): string | undefined {
		return data && typeof data === 'object' ? (data as { mode?: string }).mode : undefined;
	}

	/** Produce a compact label for a collapsed non-message event. */
	function eventLabel(event: AgentEvent): string {
		if (event.type === 'auth' && authMode(event.data) === 'api_key_fallback') {
			return 'API key fallback';
		}
		const toolName = eventToolName(event.data);
		if (toolName) return `Tool call: ${toolName}`;
		const labels: Record<string, string> = {
			reasoning: 'Reasoning',
			status: 'Status',
			item: 'Tool activity',
			command: 'Command',
			file: 'File change',
			usage: 'Token usage',
			retry: 'Retrying',
			auth: 'Authentication'
		};
		return event.method
			? `${labels[event.type] ?? event.type}: ${event.method}`
			: (labels[event.type] ?? event.type);
	}

	/** Extract a human-readable tool name from an SDK item payload. */
	function eventToolName(data: unknown): string | undefined {
		if (!data || typeof data !== 'object') return undefined;
		const record = data as Record<string, unknown>;
		const item =
			record.item && typeof record.item === 'object'
				? (record.item as Record<string, unknown>)
				: record;
		if (typeof item.type !== 'string' || !item.type.toLowerCase().includes('tool'))
			return undefined;
		for (const key of ['name', 'toolName', 'tool_name', 'serverName', 'server_name']) {
			if (typeof item[key] === 'string') return item[key];
		}
		return undefined;
	}

	onMount(initialize);
</script>

<section class="agent-page">
	<header class="agent-header">
		<div>
			<p class="eyebrow">Codex agent session</p>
			<h1>Investigate your archive</h1>
		</div>
		<div class="controls">
			{#if running}<button class="stop" onclick={stop} disabled={stopping}
					>{stopping ? 'Stopping…' : 'Stop'}</button
				>{/if}
			<button class="new-session" onclick={startNewSession} disabled={clearing || submitting}
				>{clearing ? 'Starting…' : 'New session'}</button
			>
		</div>
	</header>
	{#if error}<p class="error">{error}</p>{/if}
	<div class="transcript" aria-live="polite">
		{#if loading}<p class="status">Loading Codex thread…</p>{:else if entries.length === 0}<p
				class="status"
			>
				{running ? 'Connecting to Codex…' : 'No archived messages.'}
			</p>{/if}
		{#each entries as entry (entry.id)}
			{#if entry.kind === 'user' || entry.kind === 'assistant'}
				<article class="message {entry.kind}">
					<p class="message-label">{entry.label}</p>
					{#if entry.kind === 'assistant'}
						<div class="message-text markdown">
							<!-- eslint-disable-next-line svelte/no-at-html-tags -->
							{@html renderMarkdown(entry.text)}
						</div>
					{:else}
						<div class="message-text">{entry.text}</div>
					{/if}
				</article>
			{:else if entry.kind === 'turn-notice'}
				<p class="turn-notice">{entry.text}</p>
			{:else}
				<details class="activity {entry.kind}" class:reasoning={entry.isReasoning}>
					<summary
						>{entry.label}{#if entry.summary}: <span class="markdown"
								><!-- eslint-disable-next-line svelte/no-at-html-tags -->
								{@html renderMarkdown(entry.summary)}</span
							>{/if}</summary
					>{#if entry.text}<pre>{entry.text}</pre>{/if}{#if entry.events}<pre>{JSON.stringify(
								entry.events,
								null,
								2
							)}</pre>{/if}
				</details>
			{/if}
		{/each}
	</div>
	<form
		class="composer"
		onsubmit={(event: SubmitEvent) => {
			event.preventDefault();
			sendMessage();
		}}
	>
		<textarea
			placeholder={running ? 'Codex is working…' : 'Ask a follow-up question…'}
			bind:value={message}
			disabled={running || submitting}
			rows="2"
		></textarea>
		<button type="submit" disabled={running || submitting || !message.trim()}
			>{submitting ? 'Sending…' : 'Send'}</button
		>
	</form>
</section>

<style>
	.agent-page {
		display: flex;
		flex-direction: column;
		gap: var(--s4);
	}
	.agent-header,
	.controls,
	.composer {
		display: flex;
		align-items: center;
		gap: var(--s3);
	}
	.agent-header {
		justify-content: space-between;
		padding-bottom: var(--s4);
		border-bottom: 1px solid var(--border);
	}
	h1,
	.eyebrow {
		margin: 0;
	}
	h1 {
		font-size: 20px;
	}
	.eyebrow {
		color: var(--accent);
		font-size: 11px;
		font-weight: 600;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}
	button {
		border-radius: var(--r2);
		padding: 7px 12px;
		border: 1px solid var(--border-strong);
		background: var(--surface);
		color: var(--text);
		font: inherit;
		cursor: pointer;
	}
	button:disabled {
		cursor: default;
		opacity: 0.5;
	}
	.stop {
		color: var(--error);
		border-color: var(--error);
	}
	.transcript {
		display: flex;
		flex-direction: column;
		gap: var(--s3);
		min-height: 320px;
	}
	.message {
		max-width: 82%;
		padding: var(--s3) var(--s4);
		border: 1px solid var(--border);
		border-radius: var(--r3);
		background: var(--surface);
	}
	.message.user {
		align-self: flex-end;
		background: var(--elevated);
		border-color: var(--border-strong);
	}
	.message.assistant {
		border-left: 2px solid var(--accent);
	}
	.message-label {
		margin: 0 0 var(--s1);
		color: var(--accent);
		font-size: 11px;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}
	.message-text {
		overflow-wrap: anywhere;
	}
	.markdown :global(p:first-child) {
		margin-top: 0;
	}
	.markdown :global(p:last-child) {
		margin-bottom: 0;
	}
	.markdown :global(ul),
	.markdown :global(ol) {
		padding-left: var(--s5);
	}
	.activity {
		padding: var(--s2) var(--s3);
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--r2);
		color: var(--muted);
		font-size: 12px;
	}
	.activity summary {
		cursor: pointer;
	}
	.activity.error {
		border-color: var(--error);
		color: var(--error);
	}
	.activity.notice {
		border-color: var(--warn);
		color: var(--warn);
	}
	.activity.reasoning {
		background-color: #0d0d1a;
		border-color: #252545;
		border-left: 2px solid #4a4a8a;
		color: #8080b0;
		opacity: 0.75;
	}
	.turn-notice {
		margin: 0;
		padding: var(--s2) var(--s3);
		color: var(--warn);
		background: var(--surface);
		border: 1px solid var(--warn);
		border-radius: var(--r2);
		font-size: 12px;
	}
	.activity pre {
		margin-top: var(--s2);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.composer {
		position: sticky;
		bottom: var(--s4);
		padding: var(--s3);
		background: var(--bg);
		border: 1px solid var(--border-strong);
		border-radius: var(--r3);
	}
	.composer textarea {
		min-width: 0;
		flex: 1;
		resize: vertical;
		padding: var(--s2);
		color: var(--text);
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--r2);
		font: inherit;
	}
	.composer button {
		color: var(--bg);
		background: var(--accent);
		border-color: var(--accent);
		font-weight: 600;
	}
	@media (max-width: 640px) {
		.agent-header,
		.composer {
			align-items: stretch;
			flex-direction: column;
		}
		.message {
			max-width: 100%;
		}
	}
</style>
