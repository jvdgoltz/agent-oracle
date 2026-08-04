<script lang="ts">
	import { knownAgent, type Agent } from './format';

	interface Props {
		agent: string;
	}

	let { agent }: Props = $props();

	const colors: Record<string, string> = {
		codex: '#7aa2f7',
		factory: '#ffb86c',
		claude: '#f7768e'
	};

	const normalized: Agent | null = $derived(knownAgent(agent));
</script>

<!-- Agent badge with a distinct background per agent type. -->
<span class="badge" style={`--badge-color: ${normalized ? colors[normalized] : '#9a9a9a'}`}
	>{agent}</span
>

<style>
	.badge {
		display: inline-block;
		padding: 0.15em 0.6em;
		border-radius: 999px;
		background-color: var(--badge-color);
		color: #1a1a1a;
		font-size: 0.75rem;
		font-weight: 600;
		white-space: nowrap;
	}
</style>
