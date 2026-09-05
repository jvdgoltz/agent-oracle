/** Regression tests for archive browsing and overlapping requests. */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createBrowserState, createSessionBrowser } from './session-browser.ts';
import type { SearchResponse, SearchResult, SessionList } from './api.ts';

/** Expose promise settlement so responses can arrive in any order. */
function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (reason: Error) => void;
	const promise = new Promise<T>((done, fail) => {
		resolve = done;
		reject = fail;
	});
	return { promise, resolve, reject };
}

/** Build an isolated browser with controllable backend responses. */
function setup() {
	const feeds: { args: unknown[]; response: ReturnType<typeof deferred<SessionList>> }[] = [];
	const searches: { args: unknown[]; response: ReturnType<typeof deferred<SearchResponse>> }[] = [];
	const summaries: { args: unknown[]; response: ReturnType<typeof deferred<string>> }[] = [];
	const state = createBrowserState();
	const browser = createSessionBrowser(state, {
		getSessions: (...args: unknown[]) => {
			const response = deferred<SessionList>();
			feeds.push({ args, response });
			return response.promise;
		},
		search: (...args: unknown[]) => {
			const response = deferred<SearchResponse>();
			searches.push({ args, response });
			return response.promise;
		},
		fetchSearchSummary: (...args: unknown[]) => {
			const response = deferred<string>();
			summaries.push({ args, response });
			return response.promise;
		}
	});
	return { state, browser, feeds, searches, summaries };
}

/** Create a minimal real search-result shape. */
function result(id: string): SearchResult {
	return {
		session_id: id,
		title: id,
		agent: 'codex',
		cwd: '/',
		started_at: null,
		summary: null,
		entities: [],
		snippet: '',
		message_snippets: [],
		score: 1
	};
}

test('feed sends the filter and page offset to the server and keeps its total', async () => {
	const { state, browser, feeds } = setup();
	state.agent = 'pi';
	state.offset = 50;
	const pending = browser.loadFeed();
	assert.deepEqual(feeds[0].args.slice(0, 4), [50, 50, false, 'pi']);
	feeds[0].response.resolve({ sessions: [], total: 123 });
	await pending;
	assert.equal(state.total, 123);
	assert.equal(state.loading, false);
});

test('an older feed response cannot replace the selected agent or its loading state', async () => {
	const { state, browser, feeds } = setup();
	const older = browser.loadFeed();
	state.agent = 'claude';
	const newer = browser.loadFeed();
	assert.equal((feeds[0].args[4] as AbortSignal).aborted, true);
	feeds[0].response.resolve({ sessions: [], total: 99 });
	await older;
	assert.equal(state.loading, true);
	feeds[1].response.resolve({ sessions: [], total: 7 });
	await newer;
	assert.equal(state.total, 7);
});

test('draft typing leaves the feed visible and summary uses the submitted query', async () => {
	const { state, browser, searches, summaries } = setup();
	state.query = '  first query  ';
	assert.equal(state.submittedQuery, '');
	const pending = browser.runSearch();
	state.query = 'unsubmitted edit';
	searches[0].response.resolve({ results: [result('first')] });
	await pending;
	assert.equal(state.submittedQuery, 'first query');
	assert.equal(summaries[0].args[0], 'first query');
});

test('an older search failure cannot erase newer results', async () => {
	const { state, browser, searches } = setup();
	state.query = 'old';
	const older = browser.runSearch();
	state.query = 'new';
	const newer = browser.runSearch();
	searches[1].response.resolve({ results: [result('new')] });
	await newer;
	searches[0].response.reject(new Error('old failure'));
	await older;
	assert.equal(state.results[0].session_id, 'new');
	assert.equal(state.error, null);
});

test('a stale summary cannot replace the newer summary', async () => {
	const { state, browser, searches, summaries } = setup();
	state.query = 'old';
	const older = browser.runSearch();
	searches[0].response.resolve({ results: [result('old')] });
	await older;
	state.query = 'new';
	const newer = browser.runSearch();
	searches[1].response.resolve({ results: [result('new')] });
	await newer;
	summaries[1].response.resolve('new summary');
	await Promise.resolve();
	summaries[0].response.resolve('old summary');
	await Promise.resolve();
	assert.equal(state.aiSummary, 'new summary');
});

test('clearing cancels pending searches and returns to the first filtered feed page', async () => {
	const { state, browser, searches, feeds } = setup();
	state.query = 'old';
	state.agent = 'codex';
	state.offset = 50;
	const pending = browser.runSearch();
	const clearing = browser.clearSearch();
	assert.equal(state.query, '');
	assert.equal(state.submittedQuery, '');
	assert.equal(state.offset, 0);
	searches[0].response.resolve({ results: [result('old')] });
	await pending;
	assert.deepEqual(state.results, []);
	feeds[0].response.resolve({ sessions: [], total: 10 });
	await clearing;
	assert.equal(state.searching, false);
	assert.equal(state.total, 10);
});
