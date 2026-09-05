<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();

	/** Show a back-to-home link when we're inside a session. */
	let isSession = $derived(page.url.pathname.startsWith('/sessions/'));
</script>

<svelte:head>
	<title>Agent Oracle</title>
	<link rel="icon" href={favicon} />
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
	<link
		href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
		rel="stylesheet"
	/>
</svelte:head>

<nav class="nav" aria-label="Main navigation">
	<div class="nav-inner">
		<a class="logo" href={resolve('/')}>
			<span class="logo-icon">◈</span>
			<span class="logo-text">Agent Oracle</span>
		</a>
		<div class="stats-nav">
			<a class:active={page.url.pathname === '/stats'} href={resolve('/stats')}>Overview</a>
			<a
				class:active={page.url.pathname.startsWith('/stats/behavior')}
				href={resolve('/stats/behavior')}>User behavior</a
			>
		</div>
		{#if isSession}
			<a class="back-link" href={resolve('/')}>← All sessions</a>
		{/if}
	</div>
</nav>

<main class="layout">
	{@render children()}
</main>

<style>
	/* ── Design tokens ─────────────────────────────────────────────── */
	:global(:root) {
		/* Surfaces */
		--bg: #0d0d0f;
		--surface: #131316;
		--elevated: #1a1a1f;
		--hover: #1f1f26;

		/* Borders */
		--border: #252530;
		--border-strong: #333345;

		/* Text */
		--text: #e2e2e8;
		--muted: #9292a3;
		--subtle: #747489;

		/* Accent */
		--accent: #7aa2f7;
		--accent-dim: rgba(122, 162, 247, 0.12);
		--accent-glow: rgba(122, 162, 247, 0.06);

		/* Agent colors */
		--codex-color: #7aa2f7;
		--factory-color: #ffb86c;
		--claude-color: #e879a0;
		--omp-color: #9ece6a;

		/* Semantic */
		--error: #f7768e;
		--success: #9ece6a;
		--warn: #e0af68;

		/* Spacing scale (8px base) */
		--s1: 4px;
		--s2: 8px;
		--s3: 12px;
		--s4: 16px;
		--s5: 24px;
		--s6: 32px;
		--s7: 48px;
		--s8: 64px;

		/* Radii */
		--r1: 4px;
		--r2: 6px;
		--r3: 8px;
		--r4: 12px;

		/* Fonts */
		--font-sans: 'Inter', system-ui, -apple-system, sans-serif;
		--font-mono: 'JetBrains Mono', 'SFMono-Regular', 'Consolas', monospace;
	}

	/* ── Reset & base ──────────────────────────────────────────────── */
	:global(*) {
		box-sizing: border-box;
	}

	:global(body) {
		margin: 0;
		color: var(--text);
		background-color: var(--bg);
		font-family: var(--font-sans);
		font-size: 14px;
		line-height: 1.6;
		-webkit-font-smoothing: antialiased;
	}

	:global(a) {
		color: var(--accent);
		text-decoration: none;
	}

	:global(:focus-visible) {
		outline: 2px solid var(--accent);
		outline-offset: 3px;
	}

	:global(a:hover) {
		text-decoration: underline;
	}

	:global(.row) {
		display: flex;
		align-items: center;
		gap: var(--s2);
	}

	:global(.time) {
		font-size: 12px;
		color: var(--muted);
		font-variant-numeric: tabular-nums;
	}

	:global(.cwd) {
		font-family: var(--font-mono);
		font-size: 12px;
		color: var(--muted);
	}

	:global(.status) {
		color: var(--muted);
		font-size: 13px;
	}

	:global(.error) {
		color: var(--error);
		font-size: 13px;
	}

	/* ── Code blocks (markdown output) ────────────────────────────── */
	:global(code) {
		font-family: var(--font-mono);
		font-size: 12px;
		background-color: var(--elevated);
		border: 1px solid var(--border);
		border-radius: var(--r1);
		padding: 0.1em 0.35em;
	}

	:global(pre) {
		background-color: var(--surface) !important;
		border: 1px solid var(--border) !important;
		border-radius: var(--r3) !important;
		padding: var(--s4) !important;
		overflow-x: auto;
		font-family: var(--font-mono);
		font-size: 12px;
		line-height: 1.5;
	}

	:global(pre code) {
		background: none !important;
		border: none !important;
		padding: 0 !important;
		font-size: inherit;
	}

	/* ── Navigation ────────────────────────────────────────────────── */
	.nav {
		position: sticky;
		top: 0;
		z-index: 100;
		background-color: rgba(13, 13, 15, 0.85);
		backdrop-filter: blur(12px);
		border-bottom: 1px solid var(--border);
	}

	.nav-inner {
		max-width: 900px;
		margin: 0 auto;
		padding: 0 var(--s5);
		height: 48px;
		display: flex;
		align-items: center;
		gap: var(--s5);
	}

	.logo {
		display: flex;
		align-items: center;
		gap: var(--s2);
		color: var(--text);
		font-weight: 600;
		font-size: 14px;
		letter-spacing: -0.01em;
		text-decoration: none;
	}

	.logo:hover {
		text-decoration: none;
		color: var(--text);
	}

	.logo-icon {
		color: var(--accent);
		font-size: 16px;
		line-height: 1;
	}

	.logo-text {
		color: var(--text);
	}

	.back-link {
		font-size: 13px;
		color: var(--muted);
		transition: color 0.15s;
	}

	.stats-nav {
		margin-left: auto;
		display: flex;
		gap: var(--s3);
	}

	.stats-nav a {
		color: var(--muted);
		font-size: 12px;
	}

	.stats-nav a:hover,
	.stats-nav a.active {
		color: var(--text);
		text-decoration: none;
	}

	.back-link:hover {
		color: var(--text);
		text-decoration: none;
	}

	/* ── Page container ────────────────────────────────────────────── */
	.layout {
		max-width: 900px;
		margin: 0 auto;
		padding: var(--s6) var(--s5);
	}

	@media (max-width: 640px) {
		.nav-inner {
			height: auto;
			min-height: 48px;
			flex-wrap: wrap;
			gap: var(--s2) var(--s3);
			padding: var(--s2) var(--s4);
		}
		.back-link {
			flex-basis: 100%;
		}
		.layout {
			padding: var(--s4);
		}
	}
</style>
