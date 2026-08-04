/**
 * Formatting helpers shared across the frontend.
 */

/** The set of agents that receive distinct badge styling. */
export type Agent = 'codex' | 'factory' | 'claude';

/**
 * Format an ISO timestamp as a short relative time string (e.g. "3m ago").
 * @param iso - ISO 8601 timestamp string.
 * @returns A human-friendly relative description.
 */
export function relativeTime(iso: string): string {
	const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
	if (seconds < 60) return 'just now';
	const minutes = Math.floor(seconds / 60);
	if (minutes < 60) return `${minutes}m ago`;
	const hours = Math.floor(minutes / 60);
	if (hours < 24) return `${hours}h ago`;
	const days = Math.floor(hours / 24);
	if (days < 30) return `${days}d ago`;
	return new Date(iso).toLocaleDateString();
}

/**
 * Map an agent name to one of the known badge agents, if any.
 * @param agent - Raw agent identifier from the backend.
 * @returns The normalized agent, or null when unknown.
 */
export function knownAgent(agent: string): Agent | null {
	const normalized = agent.toLowerCase();
	if (normalized.includes('codex')) return 'codex';
	if (normalized.includes('factory') || normalized.includes('droid')) return 'factory';
	if (normalized.includes('claude')) return 'claude';
	return null;
}
