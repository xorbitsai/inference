import type { FieldFilter, HistoricalLogHandoff } from './types';

export type RuntimeLogSourceRole = 'supervisor' | 'local' | 'worker';

export interface RuntimeLogSource {
  id: string;
  label: string;
  role?: RuntimeLogSourceRole;
  node_name?: string;
}

export interface NormalizedRuntimeLogSource {
  id: string;
  label: string;
  role: RuntimeLogSourceRole;
  nodeName: string;
  displayNodeName: string;
}

export function getNodeDisplayName(nodeName: string): string {
  return nodeName.trim();
}

export function normalizeRuntimeLogSource(source: RuntimeLogSource): NormalizedRuntimeLogSource {
  const role: RuntimeLogSourceRole =
    source.role ??
    (source.id === 'supervisor' ? 'supervisor' : source.id === 'local' ? 'local' : 'worker');
  const nodeName =
    source.node_name?.trim() || (role === 'worker' ? source.id : source.label || source.id);
  return {
    id: source.id,
    label: source.label,
    role,
    nodeName,
    displayNodeName: getNodeDisplayName(nodeName),
  };
}

export function runtimeLogSourceSearchText(
  source: NormalizedRuntimeLogSource,
  roleLabel: string
): string {
  return [
    source.id,
    source.label,
    source.role,
    roleLabel,
    source.nodeName,
    source.displayNodeName,
  ].join(' ');
}

export interface RuntimeLogEntry {
  id: string;
  source: string;
  timestamp: string;
  timestampMs?: number;
  level?: string;
  message: string;
  raw: string;
  sourceSequence: number;
}

export interface HistoricalHandoffQueryState {
  searchText: string;
  appliedSearch: string;
  selectedLevels: string[];
  selectedLogType: string;
  selectedNodes: string[];
  pageFrom: number;
  fieldFilters: FieldFilter[];
}

export function buildHistoricalHandoffQueryState(
  handoff: HistoricalLogHandoff
): HistoricalHandoffQueryState {
  const query = handoff.requestId || handoff.query || '';
  return {
    searchText: query,
    appliedSearch: query,
    selectedLevels: handoff.level ? [handoff.level === 'WARN' ? 'WARNING' : handoff.level] : [],
    selectedLogType: '',
    selectedNodes: [],
    pageFrom: 0,
    fieldFilters: [],
  };
}

const TEXT_LOG_START =
  /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2}))\s+([A-Z]+)\s+(.*)$/;
const TEXT_FILE_MESSAGE = /^\S+\s+pid:\d+\s+role:\S*\s+address:\S*\s+node:\S+(?:\s+(.*))?$/;
const ENTRY_BOUNDARY = /(?:^|\n)(?=(?:\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}|\{"@timestamp"\s*:))/;

interface ParsedLogStart {
  timestamp: string;
  timestampMs?: number;
  level?: string;
  message: string;
}

function timestampMs(timestamp: string): number | undefined {
  const parsed = Date.parse(timestamp);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseLogStart(line: string): ParsedLogStart | undefined {
  if (line.startsWith('{')) {
    try {
      const value = JSON.parse(line) as Record<string, unknown>;
      const timestamp = typeof value['@timestamp'] === 'string' ? value['@timestamp'] : '';
      if (timestamp) {
        return {
          timestamp,
          timestampMs: timestampMs(timestamp),
          level: typeof value.level === 'string' ? value.level : undefined,
          // Keep the complete structured record visible and searchable.
          message: line,
        };
      }
    } catch {
      // A chunk may end in the middle of a JSON line. It will be parsed again
      // after the next chunk is appended.
    }
  }

  const match = TEXT_LOG_START.exec(line);
  if (!match) return undefined;
  return {
    timestamp: match[1],
    timestampMs: timestampMs(match[1]),
    level: match[2],
    message: match[3],
  };
}

export function parseRuntimeLogPayload(raw: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // Runtime logs are commonly plain text; an invalid JSON payload is expected.
  }
  return undefined;
}

export function getRuntimeHistoricalSearch(entry: RuntimeLogEntry): {
  requestId?: string;
  query?: string;
} {
  const payload = parseRuntimeLogPayload(entry.raw);
  const requestId = typeof payload?.request_id === 'string' ? payload.request_id.trim() : undefined;
  if (requestId) return { requestId };

  if (payload) {
    const message = typeof payload.message === 'string' ? payload.message.trim() : '';
    return message ? { query: message.slice(0, 160) } : {};
  }

  // An incomplete structured record is not a reliable Elasticsearch message
  // query. Let the node and time window locate it instead.
  if (entry.raw.trimStart().startsWith('{')) return {};

  const firstLine = entry.raw.split('\n', 1)[0];
  const logStart = TEXT_LOG_START.exec(firstLine);
  const textMessage = logStart ? TEXT_FILE_MESSAGE.exec(logStart[3])?.[1]?.trim() : undefined;
  // Only build a query when the TextFileFormatter metadata prefix can be
  // removed reliably. Filebeat stores the remaining text as `message`.
  return textMessage ? { query: textMessage.slice(0, 160) } : {};
}

export function parseRuntimeLogEntries(logs: string, source: string): RuntimeLogEntry[] {
  if (!logs) return [];

  const entries: RuntimeLogEntry[] = [];
  const lines = logs.split('\n');
  const hasTrailingNewline = logs.endsWith('\n');
  let current: RuntimeLogEntry | undefined;

  const pushCurrent = () => {
    if (!current) return;
    current.id = `${source}:${current.sourceSequence}:${current.timestamp || 'unknown'}`;
    entries.push(current);
    current = undefined;
  };

  lines.forEach((line, index) => {
    if (hasTrailingNewline && index === lines.length - 1 && line === '') return;

    const start = parseLogStart(line);
    if (start) {
      pushCurrent();
      current = {
        id: '',
        source,
        timestamp: start.timestamp,
        timestampMs: start.timestampMs,
        level: start.level,
        message: start.message,
        raw: line,
        sourceSequence: entries.length,
      };
      return;
    }

    if (current) {
      current.raw += `\n${line}`;
      current.message += `\n${line}`;
      return;
    }

    // Preserve unrecognized content rather than dropping it. This also covers
    // legacy log formats and a partial first line after an old buffer was cut.
    current = {
      id: '',
      source,
      timestamp: '',
      message: line,
      raw: line,
      sourceSequence: entries.length,
    };
  });

  pushCurrent();
  return entries;
}

export function mergeRuntimeLogEntries(
  entriesBySource: Record<string, RuntimeLogEntry[]>,
  sourceOrder: string[]
): RuntimeLogEntry[] {
  const order = new Map(sourceOrder.map((source, index) => [source, index]));
  return sourceOrder
    .flatMap((source) => entriesBySource[source] || [])
    .sort((left, right) => {
      const leftTimestamp = left.timestampMs;
      const rightTimestamp = right.timestampMs;
      if (
        leftTimestamp !== undefined &&
        rightTimestamp !== undefined &&
        leftTimestamp !== rightTimestamp
      ) {
        return leftTimestamp - rightTimestamp;
      }
      if (leftTimestamp !== undefined && rightTimestamp === undefined) return -1;
      if (leftTimestamp === undefined && rightTimestamp !== undefined) return 1;

      const sourceDifference =
        (order.get(left.source) ?? Number.MAX_SAFE_INTEGER) -
        (order.get(right.source) ?? Number.MAX_SAFE_INTEGER);
      if (sourceDifference !== 0) return sourceDifference;
      return left.sourceSequence - right.sourceSequence;
    });
}

function physicalLineCount(entry: RuntimeLogEntry): number {
  if (!entry.raw) return 1;
  return entry.raw.split('\n').length;
}

export function tailRuntimeLogEntries(
  entries: RuntimeLogEntry[],
  maxLines: number
): RuntimeLogEntry[] {
  if (entries.length === 0 || maxLines <= 0) return [];

  let lines = 0;
  let start = entries.length;
  for (let index = entries.length - 1; index >= 0; index--) {
    const entryLines = physicalLineCount(entries[index]);
    if (lines > 0 && lines + entryLines > maxLines) break;
    start = index;
    lines += entryLines;
    if (lines >= maxLines) break;
  }
  return entries.slice(start);
}

export function filterRuntimeLogEntries(
  entries: RuntimeLogEntry[],
  search: string,
  sourceSearchText: Record<string, string> = {}
): RuntimeLogEntry[] {
  const needle = search.trim().toLowerCase();
  if (!needle) return entries;
  return entries.filter((entry) =>
    [
      entry.source,
      sourceSearchText[entry.source] || '',
      entry.timestamp,
      entry.level || '',
      entry.raw,
    ]
      .join('\n')
      .toLowerCase()
      .includes(needle)
  );
}

export function trimRuntimeLogBuffer(logs: string, maxChars: number): string {
  if (logs.length <= maxChars) return logs;

  const candidate = logs.slice(logs.length - maxChars);
  const boundary = ENTRY_BOUNDARY.exec(candidate);
  if (boundary) {
    const offset = boundary.index + (boundary[0] === '\n' ? 1 : 0);
    return candidate.slice(offset);
  }

  const newline = candidate.indexOf('\n');
  return newline >= 0 ? candidate.slice(newline + 1) : candidate;
}

export function getRuntimeLogSourceColorIndex(source: string, colorCount: number): number {
  if (colorCount <= 0) return 0;
  let hash = 0;
  for (let index = 0; index < source.length; index++) {
    hash = (hash * 31 + source.charCodeAt(index)) | 0;
  }
  return Math.abs(hash) % colorCount;
}

// Kept for compatibility with existing callers and tests.
export function tailLogLines(logs: string, maxLines: number): string {
  let end = logs.endsWith('\n') ? logs.length - 1 : logs.length;

  for (let line = 0; line < maxLines; line++) {
    if (end <= 0) return logs;
    const newline = logs.lastIndexOf('\n', end - 1);
    if (newline < 0) return logs;
    end = newline;
  }

  return logs.slice(end + 1);
}
