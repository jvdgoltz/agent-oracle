<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<title>Agent Oracle</title>
	<link rel="icon" href={favicon} />
</svelte:head>

{@render children()}

<style>
	:global(*) {
		box-sizing: border-box;
	}

	:global(body) {
		margin: 0;
		color: #e6e6e6;
		background-color: #1a1a1a;
		font-family:
			'Inter',
			system-ui,
			-apple-system,
			'Segoe UI',
			Roboto,
			sans-serif;
		line-height: 1.5;
	}

	:global(a) {
		color: #7aa2f7;
		text-decoration: none;
	}

	:global(a:hover) {
		text-decoration: underline;
	}

	:global(code) {
		background-color: #2a2a2a;
		border-radius: 4px;
		padding: 0.15em 0.35em;
		font-family: 'SFMono-Regular', 'Consolas', 'Liberation Mono', monospace;
		font-size: 0.9em;
	}

	:global(.row) {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	:global(.time) {
		font-size: 0.75rem;
		color: #8a8a8a;
	}

	:global(.cwd) {
		font-family: 'SFMono-Regular', 'Consolas', monospace;
		font-size: 0.8rem;
		color: #8a8a8a;
	}

	:global(.status) {
		color: #8a8a8a;
	}

	:global(.error) {
		color: #f7768e;
	}
</style>
