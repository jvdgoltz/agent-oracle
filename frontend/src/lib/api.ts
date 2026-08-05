/**
 * Typed client for the Agent Oracle backend REST API.
 *
 * All functions talk to the backend at the configured API base URL and
 * return parsed JSON. Fetch errors are surfaced as rejected promises.
 */

/**
 * Base URL for the backend API, overridable via `VITE_API_URL`.
 */
const API_BASE_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

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
	agent: string;
	cwd: string;
	started_at: string;
	summary: string;
	entities: Entity[];
}

/** Full session detail including messages and entities. */
export interface SessionDetail {
	id: string;
	agent: string;
	cwd: string;
	started_at: string;
	messages: Message[];
	entities: Entity[];
	summary: string;
}

/** A single search hit across sessions. */
export interface SearchResult {
	session_id: string;
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

/** Valid search modes accepted by the backend. */
export type SearchMode = 'text' | 'vector' | 'hybrid';

/**
 * Perform a request against the backend and parse the JSON response.
 * @param path - API path relative to the base URL.
 * @returns The parsed JSON response body.
 */
async function get<T>(path: string): Promise<T> {
	const response = await fetch(`${API_BASE_URL}${path}`);
	if (!response.ok) {
		throw new Error(`Request failed: ${response.status} ${response.statusText}`);
	}
	return (await response.json()) as T;
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
export function getSessions(limit = 50, offset = 0): Promise<SessionList> {
	return get(`/api/sessions?limit=${limit}&offset=${offset}`);
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
export async function fetchSearchSummary(query: string, results: SearchResult[]): Promise<string> {
	const response = await fetch(`${API_BASE_URL}/api/search/summary`, {
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
	entity?: string
): Promise<SearchResponse> {
	const params = new URLSearchParams({ q, mode, limit: String(limit) });
	if (agent) params.set('agent', agent);
	if (entity) params.set('entity', entity);
	return get(`/api/search?${params.toString()}`);
}

/**
 * Fetch entities for a session.
 * @param sessionId - The session identifier.
 */
export function getEntities(sessionId: string): Promise<Entity[]> {
	return get(`/api/entities?session_id=${encodeURIComponent(sessionId)}`);
}
