/**
 * Placeholder param value baked into the static export by
 * `generateStaticParams()` for dynamic routes (see `frontend_static.py` on the
 * backend). The backend also serves this shell's prebuilt RSC flight payload
 * for any URL under the route (see the comment on `_resolve` in
 * `frontend_static.py`), so `useParams()` reflects this placeholder forever on
 * a shell-backed route, not just on first render — it must not be used to
 * read a dynamic segment's real value. Read the real value from the browser's
 * actual pathname (`usePathname()`) via `getPathSegmentsAfter` instead.
 */
export const SHELL_ROUTE_PARAM = '__shell__';

/**
 * Decode a dynamic route segment read from the URL.
 *
 * Server components receive params already decoded, but a raw URL segment
 * (e.g. from `usePathname()`) is not. Decode it so client pages see the same
 * value the old server pages did; fall back to the raw value on malformed
 * input.
 */
export function decodeRouteParam(value: string | string[] | undefined): string {
  const raw = Array.isArray(value) ? value[0] : (value ?? '');
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

/**
 * Extract the dynamic segments following a static route prefix from an actual
 * browser pathname (as returned by `usePathname()`), decoded.
 *
 * `getPathSegmentsAfter('/register-model/custom/my-model', '/register-model')`
 * -> `['custom', 'my-model']`
 */
export function getPathSegmentsAfter(pathname: string, prefix: string): string[] {
  const rest = pathname.startsWith(prefix) ? pathname.slice(prefix.length) : '';
  return rest
    .split('/')
    .filter(Boolean)
    .map((segment) => decodeRouteParam(segment));
}
