/** Rendered regression checks for behavior breakdown columns and row labels. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';

/** Compile the real table with the installed Svelte server renderer. */
async function tableComponent() {
	const source = readFileSync(new URL('./BehaviorTable.svelte', import.meta.url), 'utf8');
	const { js } = compile(source, { generate: 'server' });
	const code = js.code.replaceAll(
		"'svelte/internal/server'",
		JSON.stringify(import.meta.resolve('svelte/internal/server'))
	);
	return (await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`))
		.default;
}

/** Read text cells without Svelte hydration comments or markup. */
function cells(html: string, tag: string): string[] {
	return [...html.matchAll(new RegExp(`<${tag}\\b[^>]*>(.*?)</${tag}>`, 'gs'))].map((match) =>
		match[1].replace(/<[^>]*>/g, '').trim()
	);
}

test('behavior tables preserve all columns, rate precision, row order and escaped labels', async () => {
	const component = await tableComponent();
	const row = {
		user_messages: 123,
		yelling_rate: 1.234,
		profanity_rate: 2,
		anguish_rate: 3,
		negation_rate: 4,
		repetition_rate: 5,
		blame_rate: 6,
		interruption_rate: 7,
		detected_messages: 8,
		detection_rate: 9.876
	};
	for (const [field, label, code] of [
		['model', 'Model', true],
		['agent', 'Agent', false],
		['cwd', 'Project', true],
		['date', 'Date', false]
	] as const) {
		const { body } = render(component, {
			props: {
				rows: [
					{ ...row, [field]: '<first>' },
					{ ...row, [field]: 'second' }
				],
				field,
				label,
				code
			}
		});
		assert.deepEqual(cells(body, 'th'), [
			label,
			'Messages',
			'Yelling',
			'Profanity',
			'Anguish',
			'Negation',
			'Repetition',
			'Blame',
			'Interruptions',
			'Total',
			'Percentage'
		]);
		assert.deepEqual(cells(body, 'td'), [
			'&lt;first>',
			'123',
			'1.23%',
			'2.00%',
			'3.00%',
			'4.00%',
			'5.00%',
			'6.00%',
			'7.00%',
			'8',
			'9.88%',
			'second',
			'123',
			'1.23%',
			'2.00%',
			'3.00%',
			'4.00%',
			'5.00%',
			'6.00%',
			'7.00%',
			'8',
			'9.88%'
		]);
		assert.equal(body.includes('<code'), code);
		assert.equal((body.match(/class="percentage/g) ?? []).length, 3);
	}
	assert.deepEqual(
		cells(render(component, { props: { rows: [], field: 'model', label: 'Model' } }).body, 'td'),
		[]
	);
});
