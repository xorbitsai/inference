/**
 * Resolve the effective Token Router management capabilities.
 *
 * Advanced-auth scopes only apply after the global UI configuration is ready.
 * In no-auth mode the backend exposes the management routes anonymously, so
 * both capabilities must remain available in the UI.
 *
 * @param {{
 *   globalReady: boolean,
 *   authAdvanced?: boolean,
 *   canWriteRouters: boolean,
 *   canOperateRouters: boolean,
 * }} options
 * @returns {{ canWriteRouters: boolean, canOperateRouters: boolean }}
 */
export const resolveRouterCapabilities = ({
  globalReady,
  authAdvanced,
  canWriteRouters,
  canOperateRouters,
}) => {
  if (!globalReady || authAdvanced === undefined) {
    return { canWriteRouters: false, canOperateRouters: false };
  }
  if (authAdvanced === false) {
    return { canWriteRouters: true, canOperateRouters: true };
  }
  return { canWriteRouters, canOperateRouters };
};
