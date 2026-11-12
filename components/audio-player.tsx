"use client";

import { useLayoutEffect, useRef } from "react";

/**
 * 音声プレーヤー。
 * Next.js は別のページに移動しても前のページを隠して残すため、隠れたときに再生を止める。
 */
export function AudioPlayer({
  src,
  label,
  preload = "metadata",
}: {
  src: string;
  label: string;
  /** "none" なら再生するまで読み込まない（一覧にたくさん並ぶときに使う） */
  preload?: "metadata" | "none";
}) {
  const ref = useRef<HTMLAudioElement>(null);

  useLayoutEffect(() => {
    const audio = ref.current;
    return () => audio?.pause();
  }, []);

  return (
    <audio
      ref={ref}
      src={src}
      controls
      preload={preload}
      aria-label={label}
      className="h-10 w-full"
    />
  );
}
