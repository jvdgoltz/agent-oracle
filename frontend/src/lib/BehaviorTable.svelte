<!-- Shared signal columns for one behavior breakdown. -->
<script module lang="ts">
	import type { BehaviorSummary } from '$lib/api';

	/** Signals exposed by OMP's user-metrics classifier. */
	export const SIGNALS: {
		key: keyof BehaviorSummary;
		rate: keyof BehaviorSummary;
		label: string;
		unit: string;
	}[] = [
		{ key: 'yelling', rate: 'yelling_rate', label: 'Yelling', unit: 'hits' },
		{ key: 'profanity', rate: 'profanity_rate', label: 'Profanity', unit: 'hits' },
		{ key: 'anguish', rate: 'anguish_rate', label: 'Anguish', unit: 'hits' },
		{ key: 'negation', rate: 'negation_rate', label: 'Negation', unit: 'hits' },
		{ key: 'repetition', rate: 'repetition_rate', label: 'Repetition', unit: 'hits' },
		{ key: 'blame', rate: 'blame_rate', label: 'Blame', unit: 'hits' },
		{
			key: 'interruptions',
			rate: 'interruption_rate',
			label: 'Interruptions',
			unit: 'interruptions'
		}
	];
</script>

<script lang="ts" generics="Row extends BehaviorSummary">
	let {
		rows,
		field,
		label,
		code = false
	}: {
		rows: Row[];
		field: keyof Row;
		label: string;
		code?: boolean;
	} = $props();
</script>

<table>
	<thead>
		<tr>
			<th>{label}</th><th>Messages</th>
			{#each SIGNALS as signal (signal.key)}<th>{signal.label}</th>{/each}
			<th>Total</th><th class="percentage">Percentage</th>
		</tr>
	</thead>
	<tbody>
		{#each rows as row (row[field])}
			<tr>
				<td
					>{#if code}<code>{row[field]}</code>{:else}{row[field]}{/if}</td
				>
				<td>{row.user_messages}</td>
				{#each SIGNALS as signal (signal.key)}<td>{row[signal.rate].toFixed(2)}%</td>{/each}
				<td>{row.detected_messages}</td><td class="percentage">{row.detection_rate.toFixed(2)}%</td>
			</tr>
		{/each}
	</tbody>
</table>

<style>
	.percentage {
		font-weight: 700;
	}
</style>
