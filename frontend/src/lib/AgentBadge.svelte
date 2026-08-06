<script lang="ts">
	import { knownAgent, type Agent } from './format';

	interface Props {
		agent: string;
	}

	let { agent }: Props = $props();

	/** CSS custom property values keyed by agent name. */
	const palette: Record<Agent, { color: string; bg: string }> = {
		codex: { color: '#7aa2f7', bg: 'rgba(122, 162, 247, 0.12)' },
		factory: { color: '#ffb86c', bg: 'rgba(255, 184, 108, 0.12)' },
		claude: { color: '#e879a0', bg: 'rgba(232, 121, 160, 0.12)' },
		omp: { color: '#9ece6a', bg: 'rgba(158, 206, 106, 0.12)' }
	};

	const fallback = { color: '#6b6b80', bg: 'rgba(107, 107, 128, 0.10)' };

	const normalized: Agent | null = $derived(knownAgent(agent));
	const style = $derived(normalized ? palette[normalized] : fallback);
</script>

<!-- Agent badge: colored pill with matching background tint. -->
<span class="badge" style="--badge-color: {style.color}; --badge-bg: {style.bg}">
	{agent}
</span>

<style>
	.badge {
		display: inline-flex;
		align-items: center;
		padding: 2px 8px;
		border-radius: 999px;
		background-color: var(--badge-bg);
		border: 1px solid color-mix(in srgb, var(--badge-color) 35%, transparent);
		color: var(--badge-color);
		font-size: 11px;
		font-weight: 600;
		letter-spacing: 0.02em;
		white-space: nowrap;
		text-transform: capitalize;
	}
</style>
