/**
 * Decode a dynamic route segment read via useParams().
 *
 * Server components receive params already decoded, but useParams() returns
 * the raw URL segment. Decode it so client pages see the same value the old
 * server pages did; fall back to the raw value on malformed input.
 */
export function decodeRouteParam(value: string | string[] | undefined): string {
  const raw = Array.isArray(value) ? value[0] : (value ?? '');
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}
