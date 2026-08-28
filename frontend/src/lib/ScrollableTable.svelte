<script lang="ts">
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();
</script>

<div class="table-frame">
	<div class="scrollable-table">
		{@render children()}
	</div>
	<span class="scrollbar-header-cover" aria-hidden="true"></span>
</div>

<style>
	.table-frame {
		position: relative;
	}
	.scrollable-table {
		max-height: calc(37px * 11);
		overflow: auto;
	}
	.scrollbar-header-cover {
		position: absolute;
		top: 0;
		right: 0;
		z-index: 2;
		width: 15px;
		height: 37px;
		border-bottom: 1px solid var(--border);
		background: var(--surface);
		pointer-events: none;
	}
	.scrollable-table :global(thead) {
		position: sticky;
		top: 0;
		z-index: 1;
		background: var(--surface);
	}
	.scrollable-table :global(tr) {
		height: 37px;
	}
	.scrollable-table :global(th),
	.scrollable-table :global(td) {
		box-sizing: border-box;
		height: 37px;
		padding-block: 0;
		vertical-align: middle;
		white-space: nowrap;
	}
</style>
