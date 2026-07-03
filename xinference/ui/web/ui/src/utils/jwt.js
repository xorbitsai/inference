// JWT parsing + permission helper shared by MenuSide, PermissionGate, and
// Monitoring. Centralizing the parsing avoids three copies drifting apart
// when the JWT payload shape changes.
//
// Token lives in sessionStorage under the key `token`. The payload is the
// middle base64 segment of a standard JWT; `scopes` is an array of strings.

export function parseTokenFromSession() {
  try {
    const token = sessionStorage.getItem('token')
    if (!token) return null
    const payload = JSON.parse(atob(token.split('.')[1]))
    const scopes = Array.isArray(payload.scopes) ? payload.scopes : []
    return {
      username: payload.username || payload.sub || '',
      scopes,
    }
  } catch {
    return null
  }
}

// `admin` is a wildcard — admin users implicitly pass any scope check,
// mirroring the backend `if "admin" in token_scopes: return` shortcut in
// both auth_service modules.
export function hasPermission(scopes, scope) {
  if (!Array.isArray(scopes)) return false
  return scopes.includes(scope) || scopes.includes('admin')
}

export function isAdmin(scopes) {
  return Array.isArray(scopes) && scopes.includes('admin')
}
