"use client";

import { useId } from "react";
import { CloseIcon } from "@/components/icons";
import { Button, inputClass } from "@/components/ui";

export interface VideoRow {
  id: number;
  url: string;
  start: string;
  end: string;
}

/** サーバーの rvc_api/sources.py と合わせる */
export const MAX_VIDEOS = 10;

const TIME_RE = /^\d+(?:\.\d+)?(?::\d+(?:\.\d+)?){0,2}$/;

function toSeconds(text: string): number | null {
  const value = text.trim();
  if (!value) return null;
  return value.split(":").reduce((total, part) => total * 60 + Number(part), 0);
}

/** 入力に誤りがあればメッセージを返す（URL が空の行は対象外）。 */
export function videoRowError(row: VideoRow): string | null {
  const url = row.url.trim();
  if (!url) return null;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") throw new Error();
  } catch {
    return "URL が正しくありません（https:// から始まる URL を貼ってください）";
  }
  for (const [label, value] of [
    ["開始", row.start],
    ["終了", row.end],
  ] as const) {
    if (value.trim() && !TIME_RE.test(value.trim())) {
      return `${label}時刻の書き方が正しくありません（例: 90、1:30、1:02:30）`;
    }
  }
  const start = toSeconds(row.start);
  const end = toSeconds(row.end);
  if (start != null && end != null && end <= start) {
    return "終了時刻は、開始時刻より後にしてください";
  }
  return null;
}

export function emptyVideoRow(id: number): VideoRow {
  return { id, url: "", start: "", end: "" };
}

/** YouTube などの動画の URL を入力する欄。 */
export function VideoSources({
  rows,
  onRowsChange,
  consent,
  onConsentChange,
  disabled,
}: {
  rows: VideoRow[];
  onRowsChange: (rows: VideoRow[]) => void;
  consent: boolean;
  onConsentChange: (consent: boolean) => void;
  disabled: boolean;
}) {
  const consentId = useId();

  function update(id: number, patch: Partial<VideoRow>) {
    onRowsChange(rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  }

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        {rows.map((row, index) => {
          const error = videoRowError(row);
          return (
            <div
              key={row.id}
              className="space-y-2 rounded-xl border border-zinc-200 p-3 dark:border-zinc-800"
            >
              <div className="flex items-center gap-2">
                <input
                  type="url"
                  inputMode="url"
                  className={inputClass}
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={row.url}
                  onChange={(event) => update(row.id, { url: event.target.value })}
                  aria-label={`${index + 1}本目の動画の URL`}
                  aria-invalid={error ? true : undefined}
                  disabled={disabled}
                />
                {rows.length > 1 && (
                  <button
                    type="button"
                    onClick={() => onRowsChange(rows.filter((r) => r.id !== row.id))}
                    disabled={disabled}
                    className="shrink-0 rounded p-1.5 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
                    aria-label={`${index + 1}本目の URL を外す`}
                  >
                    <CloseIcon width={16} height={16} />
                  </button>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2 text-sm text-zinc-600 dark:text-zinc-400">
                <span>使う範囲</span>
                <input
                  className={`${inputClass} w-24`}
                  placeholder="0:00"
                  value={row.start}
                  onChange={(event) => update(row.id, { start: event.target.value })}
                  aria-label={`${index + 1}本目の開始時刻`}
                  disabled={disabled}
                />
                <span>〜</span>
                <input
                  className={`${inputClass} w-24`}
                  placeholder="最後まで"
                  value={row.end}
                  onChange={(event) => update(row.id, { end: event.target.value })}
                  aria-label={`${index + 1}本目の終了時刻`}
                  disabled={disabled}
                />
                <span className="text-xs">空欄なら動画全体を使います</span>
              </div>
              {error && <p className="text-xs text-red-700 dark:text-red-300">{error}</p>}
            </div>
          );
        })}
        {rows.length < MAX_VIDEOS && (
          <Button
            size="sm"
            variant="secondary"
            disabled={disabled}
            onClick={() =>
              onRowsChange([
                ...rows,
                emptyVideoRow(Math.max(0, ...rows.map((row) => row.id)) + 1),
              ])
            }
          >
            ＋ 動画を追加
          </Button>
        )}
      </div>

      <label
        htmlFor={consentId}
        className="flex items-start gap-2 rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-100"
      >
        <input
          id={consentId}
          type="checkbox"
          checked={consent}
          onChange={(event) => onConsentChange(event.target.checked)}
          className="mt-0.5 size-4"
          disabled={disabled}
        />
        <span>
          <span className="font-medium">
            この動画の声は自分の声、または本人から学習に使う許可をもらっています
          </span>
          <span className="mt-1 block text-xs">
            YouTube の利用規約では、許可なく動画をダウンロードすることは禁止されています。
            自分の動画や、許可を得た動画だけを使ってください。
          </span>
        </span>
      </label>

      <ul className="list-disc space-y-0.5 pl-5 text-xs text-zinc-500 dark:text-zinc-400">
        <li>その人が一人で話している（歌っている）部分の時間を指定すると、仕上がりが良くなります</li>
        <li>BGM・伴奏は、下の「音声の下ごしらえ」で取り除けます（ほかの人の声や効果音は残ることがあります）</li>
        <li>合計10〜30分ほどになるように、動画を何本か組み合わせるのがおすすめです</li>
        <li>ログインが必要な動画、メンバー限定の動画、3時間を超える動画は使えません</li>
      </ul>
    </div>
  );
}
