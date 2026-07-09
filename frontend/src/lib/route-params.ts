/**
 * Placeholder param value baked into the static export by
 * `generateStaticParams()` for dynamic routes (see `frontend_static.py` on the
 * backend). `useParams()` briefly returns this literal value on first render,
 * before Next.js hydrates and swaps in the real URL segment. Pages that fetch
 * data keyed on a dynamic param must skip that fetch while the param still
 * equals this token, or they'll hit the backend with a bogus id.
 */
export const SHELL_ROUTE_PARAM = '__shell__';

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
