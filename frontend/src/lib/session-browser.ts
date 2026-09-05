/** Archive browsing state and cancellation for feed, search, and summary requests. */
import type { SearchMode, SearchResult, SessionSummary } from './api';

/** Create state that the page can wrap in a reactive proxy. */
export function createBrowserState() {
	return {
		query: '',
		submittedQuery: '',
		mode: 'hybrid' as SearchMode,
		agent: '',
		offset: 0,
		total: 0,
		sessions: [] as SessionSummary[],
		results: [] as SearchResult[],
		loading: true,
		searching: false,
		aiSummary: '',
		summaryLoading: false,
		error: null as string | null
	};
}

/** Keep only the latest requested feed or search, including its summary. */
export function createSessionBrowser(
	state: ReturnType<typeof createBrowserState>,
	api: Pick<typeof import('./api'), 'getSessions' | 'search' | 'fetchSearchSummary'>
) {
	const pageSize = 50;
	let request: AbortController | undefined;

	/** Cancel all work belonging to the previous view. */
	function cancel() {
		request?.abort();
	}

	/** Begin a new view and clear feedback from the previous request. */
	function begin() {
		cancel();
		request = new AbortController();
		state.error = null;
		state.aiSummary = '';
		state.summaryLoading = false;
		return request.signal;
	}

	/** Fetch one page with filtering and totals computed by the server. */
	async function loadFeed() {
		const signal = begin();
		state.loading = true;
		try {
			const data = await api.getSessions(
				pageSize,
				state.offset,
				false,
				state.agent || undefined,
				signal
			);
			if (signal.aborted) return;
			state.sessions = data.sessions;
			state.total = data.total;
		} catch (error) {
			if (!signal.aborted)
				state.error = error instanceof Error ? error.message : 'Failed to load sessions';
		} finally {
			if (!signal.aborted) state.loading = false;
		}
	}

	/** Summarize exactly the submitted query and its results. */
	async function loadSummary(query: string, results: SearchResult[], signal: AbortSignal) {
		state.summaryLoading = true;
		try {
			const summary = await api.fetchSearchSummary(query, results, signal);
			if (!signal.aborted) state.aiSummary = summary;
		} catch {
			// Summaries are optional; the matching sessions remain available.
		} finally {
			if (!signal.aborted) state.summaryLoading = false;
		}
	}

	/** Return to the beginning of the selected agent's archive. */
	async function clearSearch() {
		state.query = '';
		state.submittedQuery = '';
		state.results = [];
		state.searching = false;
		state.offset = 0;
		await loadFeed();
	}

	/** Submit a query while keeping later edits separate from its results. */
	async function runSearch(query = state.query) {
		const submitted = query.trim();
		if (!submitted) return clearSearch();
		const signal = begin();
		state.submittedQuery = submitted;
		state.searching = true;
		state.results = [];
		try {
			const data = await api.search(
				submitted,
				state.mode,
				20,
				state.agent || undefined,
				undefined,
				signal
			);
			if (signal.aborted) return;
			state.results = data.results;
			if (data.results.length) void loadSummary(submitted, data.results, signal);
		} catch (error) {
			if (!signal.aborted) state.error = error instanceof Error ? error.message : 'Search failed';
		} finally {
			if (!signal.aborted) state.searching = false;
		}
	}

	/** Apply the selected filter to the submitted search or first feed page. */
	function changeAgent(agent: string) {
		if (state.agent === agent) return;
		state.agent = agent;
		state.offset = 0;
		return state.submittedQuery ? runSearch(state.submittedQuery) : loadFeed();
	}

	/** Re-run the submitted query when its retrieval mode changes. */
	function changeMode(mode: SearchMode) {
		if (state.mode === mode) return;
		state.mode = mode;
		if (state.submittedQuery) return runSearch(state.submittedQuery);
	}

	/** Move within the filtered archive using the server's total. */
	function changePage(direction: -1 | 1) {
		const offset = state.offset + direction * pageSize;
		if (state.loading || offset < 0 || offset >= state.total) return;
		state.offset = offset;
		return loadFeed();
	}

	return {
		loadFeed,
		runSearch,
		clearSearch,
		changeAgent,
		changeMode,
		changePage,
		cancel,
		pageSize
	};
}
