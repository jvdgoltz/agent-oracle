/**
 * Typed client for the Agent Oracle backend REST API.
 *
 * All functions talk to the backend at the configured API base URL and
 * return parsed JSON. Fetch errors are surfaced as rejected promises.
 */

/**
 * Base URL for the backend API, overridable via `VITE_API_URL`.
 */
const API_BASE_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8731';

/** A live Codex agent thread owned by the Agent Oracle backend. */
export interface AgentSession {
	thread_id: string;
}

/** One structured event emitted during a Codex agent turn. */
export interface AgentEvent {
	type: string;
	data: unknown;
	method?: string;
}

/** Repository-scoped archived Codex sessions eligible for resume. */
export interface ResumableAgentSessions {
	sessions: SessionSummary[];
}

/** A single message within a session. */
export interface Message {
	id: number;
	session_id: string;
	role: string;
	content: string;
	timestamp: string;
	seq: number;
	is_thinking: number;
	model: string | null;
	is_system_instruction: number;
	is_injected: number;
}

/** An extracted entity (e.g. a package, file, or command) from a session. */
interface Entity {
	type: string;
	value: string;
}

/** Summary record returned when listing sessions. */
export interface SessionSummary {
	id: string;
	title: string | null;
	agent: string;
	cwd: string;
	started_at: string;
	summary: string;
	entities: Entity[];
	review_sessions: ReviewSessionSummary[];
}

/** A Codex review session linked to its parent thread. */
export interface ReviewSessionSummary {
	id: string;
	agent: string;
	cwd: string;
	started_at: string;
	summary: string | null;
	parent_thread_id: string;
}

/** Full session detail including messages and entities. */
export interface SessionDetail {
	id: string;
	title: string | null;
	agent: string;
	cwd: string;
	started_at: string;
	parent_thread_id: string | null;
	messages: Message[];
	entities: Entity[];
	summary: string;
	review_sessions: ReviewSessionSummary[];
}

/** A single search hit across sessions. */
export interface SearchResult {
	session_id: string;
	title: string | null;
	agent: string | null;
	cwd: string | null;
	started_at: string | null;
	summary: string | null;
	entities: Entity[];
	snippet: string;
	message_snippets: string[];
	score: number;
}

/** Response shape from `GET /api/sessions`. */
export interface SessionList {
	sessions: SessionSummary[];
	total: number;
}

/** OMP-compatible counts and per-user-message rates for one scope. */
export interface BehaviorSummary {
	user_messages: number;
	interruptions: number;
	interruption_rate: number;
	detected_messages: number;
	detection_rate: number;
	chars: number;
	words: number;
	yelling: number;
	profanity: number;
	anguish: number;
	negation: number;
	repetition: number;
	blame: number;
	yelling_rate: number;
	profanity_rate: number;
	anguish_rate: number;
	negation_rate: number;
	repetition_rate: number;
	blame_rate: number;
}

/** OMP-compatible behavior statistics for the selected archive scope. */
export interface BehaviorReport {
	totals: BehaviorSummary;
	daily: (BehaviorSummary & { date: string })[];
	agents: (BehaviorSummary & { agent: string })[];
	projects: (BehaviorSummary & { cwd: string })[];
	models: (BehaviorSummary & { model: string })[];
}

/** Aggregate archive counts and duration statistics. */
export interface OverviewTotals {
	sessions: number;
	conversation_messages: number;
	assistant_messages: number;
	average_session_messages: number;
	median_session_messages: number;
	average_session_duration_seconds: number;
	median_session_duration_seconds: number;
}

/** Query-time archive overview statistics. */
export interface OverviewReport {
	totals: OverviewTotals;
	entities: { entity_type: string; entity_value: string; sessions: number }[];
	models: { model: string; messages: number }[];
	agents: { agent: string; sessions: number }[];
	projects: { cwd: string; sessions: number }[];
	session_lengths: {
		session_id: string;
		agent: string;
		cwd: string;
		messages: number;
		duration_seconds: number;
	}[];
}

/** Token totals reported by one agent/model grouping. */
export interface TokenUsageRow {
	agent: string | null;
	model: string | null;
	responses: number;
	input_tokens: number | null;
	output_tokens: number | null;
	cached_input_tokens: number | null;
	cache_creation_input_tokens: number | null;
	cache_read_input_tokens: number | null;
	reasoning_output_tokens: number | null;
	total_tokens: number | null;
	cache_hit_rate: number | null;
}

/** Token statistics for the selected archive scope. */
export interface TokenUsageReport {
	agent_model: TokenUsageRow[];
	agents: TokenUsageRow[];
	models: TokenUsageRow[];
}

/** Valid search modes accepted by the backend. */
export type SearchMode = 'text' | 'vector' | 'hybrid';

/** Fetch provider-reported token totals for the selected scope. */
export function getTokenUsageStats(
	agent?: string,
	start?: string,
	end?: string
): Promise<TokenUsageReport> {
	const params = new URLSearchParams();
	if (agent) params.set('agent', agent);
	if (start) params.set('start', start);
	if (end) params.set('end', end);
	return get(`/api/stats/tokens?${params}`);
}

/**
 * Perform a request against the backend and parse the JSON response.
 * @param path - API path relative to the base URL.
 * @returns The parsed JSON response body.
 */
async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
	const response = await fetch(`${API_BASE_URL}${path}`, { signal });
	if (!response.ok) {
		throw await responseError(response);
	}
	return (await response.json()) as T;
}

/** Send JSON to the backend and parse its response. */
async function post<T>(path: string, body?: unknown): Promise<T> {
	const response = await fetch(`${API_BASE_URL}${path}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: body === undefined ? undefined : JSON.stringify(body)
	});
	if (!response.ok) throw await responseError(response);
	return (await response.json()) as T;
}

/** Convert a FastAPI error response into a useful client-facing error. */
async function responseError(response: Response): Promise<Error> {
	const fallback = `Request failed: ${response.status} ${response.statusText}`;
	try {
		const body = (await response.json()) as { detail?: unknown };
		if (typeof body.detail === 'string') return new Error(body.detail);
		if (body.detail !== undefined) return new Error(JSON.stringify(body.detail));
	} catch {
		return new Error(fallback);
	}
	return new Error(fallback);
}

/** Start a new Codex agent thread, or resume an existing Codex thread. */
export function startAgentSession(
	message: string,
	resumeThreadId?: string,
	imageDataUrl?: string
): Promise<AgentSession> {
	return post('/api/agent/sessions', {
		message,
		...(resumeThreadId ? { resume_thread_id: resumeThreadId } : {}),
		...(imageDataUrl ? { image_data_url: imageDataUrl } : {})
	});
}

/** List archived Codex sessions from this repository that Codex can resume. */
export function getResumableAgentSessions(): Promise<ResumableAgentSessions> {
	return get('/api/agent/sessions');
}

/** Re-index and enrich one archived session from its source file. */
export function enrichSession(sessionId: string): Promise<{ status: string }> {
	return post(`/api/sessions/${encodeURIComponent(sessionId)}/enrich`);
}

/** Load an archived Codex transcript without starting a turn. */
export function getArchivedAgentSession(threadId: string): Promise<SessionDetail> {
	return get(`/api/agent/sessions/${encodeURIComponent(threadId)}`);
}

/** Send a follow-up message to an idle Codex agent thread. */
export function sendAgentMessage(
	threadId: string,
	message: string,
	imageDataUrl?: string
): Promise<AgentSession> {
	return post(`/api/agent/sessions/${encodeURIComponent(threadId)}/messages`, {
		message,
		...(imageDataUrl ? { image_data_url: imageDataUrl } : {})
	});
}

/** Stop the active Codex agent turn. */
export async function stopAgentSession(threadId: string): Promise<void> {
	const response = await fetch(
		`${API_BASE_URL}/api/agent/sessions/${encodeURIComponent(threadId)}/stop`,
		{ method: 'POST' }
	);
	if (!response.ok) throw await responseError(response);
}

/** Release the current Codex session. */
export async function deleteAgentSession(threadId: string): Promise<void> {
	const response = await fetch(
		`${API_BASE_URL}/api/agent/sessions/${encodeURIComponent(threadId)}`,
		{ method: 'DELETE' }
	);
	if (!response.ok) throw await responseError(response);
}

/** Return the SSE endpoint for the current agent turn. */
export function agentEventsUrl(threadId: string): string {
	return `${API_BASE_URL}/api/agent/sessions/${encodeURIComponent(threadId)}/events`;
}

/**
 * Check backend health.
 */
export function getHealth(): Promise<{ status: string }> {
	return get('/api/health');
}

/**
 * List recent sessions, newest first.
 * @param limit - Maximum number of sessions to return.
 * @param offset - Number of sessions to skip (for pagination).
 */
export function getSessions(
	limit = 50,
	offset = 0,
	includeReviewAgents = false,
	agent?: string,
	signal?: AbortSignal
): Promise<SessionList> {
	const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
	if (includeReviewAgents) params.set('include_review_agents', 'true');
	if (agent) params.set('agent', agent);
	return get(`/api/sessions?${params.toString()}`, signal);
}

/** Fetch OMP-compatible behavior statistics for an optional archive scope. */
export function getBehaviorStats(
	agent?: string,
	start?: string,
	end?: string
): Promise<BehaviorReport> {
	const params = new URLSearchParams();
	if (agent) params.set('agent', agent);
	if (start) params.set('start', start);
	if (end) params.set('end', end);
	const query = params.size ? `?${params.toString()}` : '';
	return get(`/api/stats/behavior${query}`);
}

/** Fetch archive overview statistics for an optional bounded scope. */
export function getOverviewStats(
	agent?: string,
	start?: string,
	end?: string
): Promise<OverviewReport> {
	const params = new URLSearchParams();
	if (agent) params.set('agent', agent);
	if (start) params.set('start', start);
	if (end) params.set('end', end);
	const query = params.size ? `?${params.toString()}` : '';
	return get(`/api/stats/overview${query}`);
}

/**
 * Fetch a single session's full detail (messages + entities).
 * @param id - The session identifier.
 */
export function getSession(id: string): Promise<SessionDetail> {
	return get(`/api/sessions/${encodeURIComponent(id)}`);
}

/** Response shape from `GET /api/search`. */
export interface SearchResponse {
	results: SearchResult[];
}

/**
 * Fetch an AI-generated summary of search results for a query.
 * @param query - The original search query.
 * @param results - The search results to summarize.
 * @returns The AI summary text, or empty string if unavailable.
 */
export async function fetchSearchSummary(
	query: string,
	results: SearchResult[],
	signal?: AbortSignal
): Promise<string> {
	const response = await fetch(`${API_BASE_URL}/api/search/summary`, {
		signal,
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ query, results })
	});
	if (!response.ok) {
		throw new Error(`Request failed: ${response.status} ${response.statusText}`);
	}
	const data = (await response.json()) as { summary: string };
	return data.summary;
}

/**
 * Search sessions across their text/vector/hybrid indexes.
 * @param q - Search query text.
 * @param mode - Which index to query.
 * @param limit - Maximum number of results to return.
 * @param agent - Optional agent filter.
 * @param entity - Optional entity filter.
 */
export function search(
	q: string,
	mode: SearchMode = 'hybrid',
	limit = 20,
	agent?: string,
	entity?: string,
	signal?: AbortSignal
): Promise<SearchResponse> {
	const params = new URLSearchParams({ q, mode, limit: String(limit) });
	if (agent) params.set('agent', agent);
	if (entity) params.set('entity', entity);
	return get(`/api/search?${params.toString()}`, signal);
}

/**
 * Fetch entities for a session.
 * @param sessionId - The session identifier.
 */
export function getEntities(sessionId: string): Promise<Entity[]> {
	return get(`/api/entities?session_id=${encodeURIComponent(sessionId)}`);
}
