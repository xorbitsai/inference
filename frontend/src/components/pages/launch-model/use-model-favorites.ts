'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useGlobal } from '@/contexts/global-context';
import { COLLECTION_STORAGE_KEY } from '@/constants/launch';
import { getAccessToken } from '@/lib/auth-storage';
import { decodeJwtPayload } from '@/lib/utils';
import request from '@/lib/request';

const endpoint = '/v1/model_favorites';

function browserFavorites(): string[] {
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(COLLECTION_STORAGE_KEY) || '[]');
    return Array.isArray(value)
      ? [...new Set(value.filter((name): name is string => typeof name === 'string' && !!name))]
      : [];
  } catch {
    return [];
  }
}

export function useModelFavorites() {
  const { globalReady, clusterAuth } = useGlobal();
  const scope = useCallback(() => {
    if (!globalReady || clusterAuth?.auth == null) return null;
    if (clusterAuth.auth === false) return 'anonymous';
    const token = getAccessToken();
    if (!token) return null;
    const id = decodeJwtPayload(token)?.user_id;
    return id != null ? `user:${id}` : token;
  }, [globalReady, clusterAuth?.auth]);
  const [state, setState] = useState({
    scope: null as string | null,
    names: [] as string[],
    ready: false,
  });
  const [pending, setPending] = useState<string[]>([]);
  const [legacy, setLegacy] = useState<string[]>([]);
  const [importing, setImporting] = useState(false);
  const generation = useRef(0);
  const activeScope = useRef<string | null>(null);
  const pendingNames = useRef(new Set<string>());
  const importActive = useRef(false);

  useEffect(() => {
    setLegacy(browserFavorites());
    let disposed = false;
    const refresh = async () => {
      const current = scope();
      if (current === activeScope.current && (pendingNames.current.size || importActive.current))
        return;
      const version = ++generation.current;
      activeScope.current = current;
      pendingNames.current.clear();
      importActive.current = false;
      setPending([]);
      setImporting(false);
      setState((previous) => ({
        scope: current,
        names: previous.scope === current ? previous.names : [],
        ready: false,
      }));
      if (!current) return;
      try {
        const names = await request.get<string[]>(endpoint);
        if (!Array.isArray(names) || names.some((name) => typeof name !== 'string')) return;
        if (!disposed && generation.current === version && scope() === current) {
          setState({ scope: current, names, ready: true });
        }
      } catch {
        // The request client reports the error. Retry on focus; never publish
        // unscoped browser data or accept writes after a failed initial read.
      }
    };
    const refreshAfterAuth = () => {
      void refresh();
    };
    void refresh();
    window.addEventListener('focus', refreshAfterAuth);
    window.addEventListener('xinference-auth-change', refreshAfterAuth);
    return () => {
      disposed = true;
      generation.current++;
      activeScope.current = null;
      window.removeEventListener('focus', refreshAfterAuth);
      window.removeEventListener('xinference-auth-change', refreshAfterAuth);
    };
  }, [scope]);

  const save = async (name: string, favorite: boolean) => {
    const current = scope();
    const version = generation.current;
    if (!current || state.scope !== current || !state.ready || pendingNames.current.has(name))
      return false;
    pendingNames.current.add(name);
    setPending([...pendingNames.current]);
    try {
      if (favorite) await request.put(endpoint, { model_name: name });
      else await request.delete(endpoint, { data: { model_name: name } });
      if (generation.current !== version || scope() !== current) return false;
      setState((previous) =>
        previous.scope !== current || generation.current !== version
          ? previous
          : {
              ...previous,
              names: favorite
                ? [...new Set([...previous.names, name])]
                : previous.names.filter((n) => n !== name),
            }
      );
      return true;
    } catch {
      return false;
    } finally {
      if (generation.current === version) {
        pendingNames.current.delete(name);
        setPending([...pendingNames.current]);
      }
    }
  };

  const importBrowserFavorites = async () => {
    if (importActive.current || !state.ready || state.scope !== scope()) return;
    const version = generation.current;
    const names = browserFavorites();
    importActive.current = true;
    setImporting(true);
    try {
      const raw = window.localStorage.getItem(COLLECTION_STORAGE_KEY);
      for (const name of names) {
        if (generation.current !== version || !(await save(name, true))) return;
      }
      if (
        generation.current === version &&
        window.localStorage.getItem(COLLECTION_STORAGE_KEY) === raw
      ) {
        window.localStorage.removeItem(COLLECTION_STORAGE_KEY);
        setLegacy([]);
      }
    } catch {
      // Storage can be unavailable even when server writes succeeded. Keep the
      // browser source for a later, idempotent retry.
    } finally {
      if (generation.current === version) {
        importActive.current = false;
        setImporting(false);
      }
    }
  };

  const current = scope();
  return {
    favorites: state.scope === current ? state.names : [],
    ready: state.ready && state.scope === current,
    pending,
    importing,
    hasBrowserFavorites: legacy.length > 0,
    toggle: (name: string) => save(name, !state.names.includes(name)),
    importBrowserFavorites,
  };
}
