import { Marked, type RendererObject } from 'marked';

/** Escape untrusted text for safe interpolation into HTML attributes. */
function escapeAttribute(value: string): string {
	return value.replace(/[&<>"]/g, (character) => {
		const entities: Record<string, string> = {
			'&': '&amp;',
			'<': '&lt;',
			'>': '&gt;',
			'"': '&quot;'
		};
		return entities[character];
	});
}

/** Return a URL only when it uses an allowed web or email protocol. */
function safeUrl(value: string): string | null {
	try {
		const url = new URL(value, 'https://agent-oracle.local');
		return ['http:', 'https:', 'mailto:'].includes(url.protocol) ? value : null;
	} catch {
		return null;
	}
}

/** Render Markdown without allowing raw HTML or unsafe resource URLs. */
export function renderMarkdown(content: string): string {
	const renderer: RendererObject = {
		html: () => '',
		link({ href, title, tokens }) {
			const text = this.parser.parseInline(tokens);
			const url = safeUrl(href);
			if (!url) return text;
			const titleAttribute = title ? ` title="${escapeAttribute(title)}"` : '';
			return `<a href="${escapeAttribute(url)}"${titleAttribute}>${text}</a>`;
		},
		image({ href, title, text }) {
			const url = safeUrl(href);
			if (!url) return escapeAttribute(text);
			const titleAttribute = title ? ` title="${escapeAttribute(title)}"` : '';
			return `<img src="${escapeAttribute(url)}" alt="${escapeAttribute(text)}"${titleAttribute}>`;
		}
	};
	return new Marked({ renderer }).parse(content, { async: false }) as string;
}
