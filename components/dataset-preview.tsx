"use client";

import { useState } from "react";
import { api, fileUrls, type DatasetFile, type DatasetStage } from "@/lib/api";
import { formatDuration } from "@/lib/format";
import { AudioPlayer } from "./audio-player";

/** 一度に並べる音声の数（多すぎると重くなるため） */
const MAX_FILES = 10;

const STAGE_LABELS: Record<DatasetStage, string> = {
  original: "下ごしらえの前",
  vocals: "BGM・伴奏を除去した後",
  dereverb: "残響を除去した後",
  karaoke: "ハモリを除去した後",
};

/**
 * 学習用の音声を聴く欄（BGM・伴奏がきちんと取れているかの確認など）。
 * 下ごしらえをした音声は、手順ごとの途中経過も聴き比べられる（どの手順で音が悪くなったかがわかる）。
 * 開いたときに一覧を読み込む。
 */
export function DatasetPreview({ model, title }: { model: string; title: string }) {
  const [files, setFiles] = useState<DatasetFile[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const detail = await api.model(model);
      setFiles(detail.dataset ?? []);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  return (
    <details
      className="text-sm"
      onToggle={(event) => {
        if (event.currentTarget.open && files === null) void load();
      }}
    >
      <summary className="cursor-pointer text-zinc-500 select-none hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200">
        {title}
      </summary>
      <div className="mt-2 space-y-3">
        {error && <p className="text-red-700 dark:text-red-300">{error}</p>}
        {files === null && !error && (
          <p className="text-zinc-500 dark:text-zinc-400">読み込んでいます…</p>
        )}
        {files?.length === 0 && (
          <p className="text-zinc-500 dark:text-zinc-400">学習用の音声が見つかりません。</p>
        )}
        {files?.slice(0, MAX_FILES).map((file) => (
          <DatasetItem key={file.file} model={model} file={file} />
        ))}
        {files && files.length > MAX_FILES && (
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            ほかに {files.length - MAX_FILES}個あります。
          </p>
        )}
      </div>
    </details>
  );
}

function DatasetItem({ model, file }: { model: string; file: DatasetFile }) {
  const stages = file.stages ?? [];
  // 最後の手順の結果 ＝ 学習に使う音声なので、それ以外の途中経過を選べるようにする
  const [stage, setStage] = useState<DatasetStage | "final">("final");
  const options: (DatasetStage | "final")[] = stages.length ? [...stages.slice(0, -1), "final"] : ["final"];
  const finalLabel = stages.length ? `${STAGE_LABELS[stages.at(-1) as DatasetStage]}（学習に使う音声）` : "学習に使う音声";
  const src = stage === "final" ? fileUrls.datasetFile(model, file.file) : fileUrls.datasetStage(model, file.file, stage);

  return (
    <div className="space-y-1.5">
      <p className="truncate text-xs text-zinc-500 dark:text-zinc-400" title={file.file}>
        {file.file}・{formatDuration(file.duration)}
      </p>
      {options.length > 1 && (
        <div role="group" aria-label={`${file.file} のどの段階を聴くか`} className="flex flex-wrap gap-1.5">
          {options.map((option) => (
            <button
              key={option}
              type="button"
              aria-pressed={stage === option}
              onClick={() => setStage(option)}
              className={`rounded-md border px-2 py-1 text-xs transition-colors ${
                stage === option
                  ? "border-violet-400 bg-violet-50 font-medium text-violet-900 dark:border-violet-500/60 dark:bg-violet-500/10 dark:text-violet-100"
                  : "border-zinc-200 text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              }`}
            >
              {option === "final" ? finalLabel : STAGE_LABELS[option]}
            </button>
          ))}
        </div>
      )}
      <AudioPlayer key={stage} src={src} label={`${file.file}（${stage === "final" ? finalLabel : STAGE_LABELS[stage]}）を再生`} />
    </div>
  );
}
