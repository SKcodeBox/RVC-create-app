"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

/**
 * データを取得し、intervalMs ごとに取り直す。intervalMs が null なら1回だけ取得する。
 * ページが非表示になると（Next.js の Activity）自動で止まり、表示されると再開する。
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number | null,
  key = "",
) {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<Error | null>(null);
  const fetcherRef = useRef(fetcher);

  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  const refresh = useCallback(async () => {
    try {
      const value = await fetcherRef.current();
      setData(value);
      setError(null);
      return value;
    } catch (caught) {
      setError(caught instanceof Error ? caught : new Error(String(caught)));
      return undefined;
    }
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      await refresh();
      if (!stopped && intervalMs) timer = setTimeout(tick, intervalMs);
    };
    void tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [refresh, intervalMs, key]);

  return { data, error, refresh };
}

function subscribeClock(callback: () => void) {
  const id = setInterval(callback, 1000);
  return () => clearInterval(id);
}

/** 1秒ごとに更新される現在時刻（UNIX 秒）。サーバーでの描画中は 0。 */
export function useNow(): number {
  return useSyncExternalStore(
    subscribeClock,
    () => Math.floor(Date.now() / 1000),
    () => 0,
  );
}
