/**
 * Parse a Xinference/xoscar actor address.
 *
 * Xoscar emits actor addresses as `host:port` and splits them at the final
 * colon, including for unbracketed IPv6 hosts. Bracketed IPv6 input is also
 * accepted because API consumers may normalize addresses before returning
 * them to the UI.
 *
 * @param {string} address
 * @returns {{ host: string, port?: string }}
 */
export const parseActorAddress = (address) => {
  const normalized = address.trim();
  if (normalized.startsWith('[')) {
    const closingBracket = normalized.indexOf(']');
    if (closingBracket >= 0) {
      const host = normalized.slice(1, closingBracket);
      const suffix = normalized.slice(closingBracket + 1);
      const port = suffix.startsWith(':') ? suffix.slice(1) : '';
      return {
        host,
        port: /^\d+$/.test(port) ? port : undefined,
      };
    }
  }

  const separator = normalized.lastIndexOf(':');
  if (separator >= 0) {
    const port = normalized.slice(separator + 1);
    if (/^\d+$/.test(port)) {
      return { host: normalized.slice(0, separator), port };
    }
  }
  return { host: normalized };
};

/**
 * @param {string} host
 * @param {string} port
 * @returns {string}
 */
const formatActorAddress = (host, port) => {
  const normalizedHost = host.replace(/^\[|\]$/g, '');
  return normalizedHost.includes(':') ? `[${normalizedHost}]:${port}` : `${normalizedHost}:${port}`;
};

/**
 * Replace the model subprocess host with the Worker host while preserving the
 * subprocess port. This makes the runtime address reachable from the UI.
 *
 * @param {{ worker_address?: string | null, model_address?: string | null }} replica
 * @returns {string | undefined}
 */
export const formatReplicaRuntimeAddress = (replica) => {
  const workerAddress = replica.worker_address?.trim();
  const modelAddress = replica.model_address?.trim();
  if (!modelAddress) return undefined;

  const parsedModelAddress = parseActorAddress(modelAddress);
  const workerHost = workerAddress
    ? parseActorAddress(workerAddress).host
    : parsedModelAddress.host;
  if (parsedModelAddress.port && workerHost) {
    return formatActorAddress(workerHost, parsedModelAddress.port);
  }
  return modelAddress;
};
