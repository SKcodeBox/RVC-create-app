"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { api, fileUrls, isActive, type ConvertJob } from "@/lib/api";
import { formatDate, formatPitch } from "@/lib/format";
import { usePolling } from "@/lib/hooks";
import { AudioPlayer } from "@/components/audio-player";
import { ConfirmButton } from "@/components/confirm-button";
import { DownloadIcon } from "@/components/icons";
import { JobProgress, StatusBadge } from "@/components/job-progress";
import { buttonClass, Card, EmptyState, SectionTitle } from "@/components/ui";
import { ConvertForm } from "./convert-form";

export function ConvertWorkspace() {
  const searchParams = useSearchParams();
  const initialModel = searchParams.get("model");
  const initialWeight = searchParams.get("weight");
  const { data: models } = usePolling(api.models, 10000);
  const { data: history, refresh: refreshHistory } = usePolling(
    () => api.jobs("convert", 30) as Promise<ConvertJob[]>,
    4000,
  );
  const [current, setCurrent] = useState<ConvertJob | null>(null);
  const currentRef = useRef<HTMLElement>(null);
  const currentId = current?.id;

  // 変換を始めたら、進み具合の欄が見えるところまで移動する（フォームが長く、押した場所からは見えないため）
  useEffect(() => {
    if (!currentId) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    currentRef.current?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "nearest" });
  }, [currentId]);

  const usable = (models ?? []).filter((model) => model.weights.length > 0);
  const others = (history ?? []).filter((job) => job.id !== current?.id);

  return (
    <div className="space-y-6">
      {models && usable.length === 0 ? (
        <Card>
          <EmptyState>
            使えるモデルがまだありません。
            <br />
            <Link href="/train" className="font-medium text-violet-700 hover:underline dark:text-violet-300">
              声のモデルを作る
            </Link>
            か、
            <Link href="/models" className="font-medium text-violet-700 hover:underline dark:text-violet-300">
              モデル一覧
            </Link>
            から他で作ったモデルを取り込んでください。
          </EmptyState>
        </Card>
      ) : (
        <ConvertForm
          models={usable}
          initialModel={initialModel}
          initialWeight={initialWeight}
          onStarted={(job) => {
            setCurrent(job);
            void refreshHistory();
          }}
        />
      )}

      {current && (
        <Card aria-labelledby="convert-now" ref={currentRef} className="scroll-mt-20">
          <SectionTitle>
            <span id="convert-now">変換の状況</span>
          </SectionTitle>
          <ConvertSummary job={current} />
          <div className="mt-3">
            <JobProgress
              key={current.id}
              initialJob={current}
              onFinished={(job) => {
                setCurrent(job as ConvertJob);
                void refreshHistory();
              }}
            />
          </div>
          {current.status === "done" && <ConvertResult job={current} />}
        </Card>
      )}

      <Card aria-labelledby="convert-history">
        <SectionTitle>
          <span id="convert-history">変換の履歴</span>
        </SectionTitle>
        {others.length === 0 ? (
          <EmptyState>まだ変換したことがありません。</EmptyState>
        ) : (
          <ul className="divide-y divide-zinc-200 dark:divide-zinc-800">
            {others.map((job, position) => (
              <li key={job.id} className="space-y-2 py-4 first:pt-0 last:pb-0">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <StatusBadge status={job.status} />
                  <ConvertSummary job={job} />
                  {!isActive(job.status) && (
                    <span className="ml-auto">
                      <ConfirmButton
                        question="この履歴と音声を削除しますか？"
                        confirmLabel="削除する"
                        onConfirm={async () => {
                          await api.deleteJob(job.id);
                          await refreshHistory();
                        }}
                      >
                        削除
                      </ConfirmButton>
                    </span>
                  )}
                </div>
                {/* 履歴がたくさんあっても重くならないよう、新しいもの以外は再生するまで読み込まない */}
                {job.status === "done" && <ConvertResult job={job} compact preload={position < 3 ? "metadata" : "none"} />}
                {job.status === "error" && job.message && (
                  <p className="text-sm text-red-700 dark:text-red-300">{job.message}</p>
                )}
                {(job.status === "error" || job.status === "interrupted") && <HistoryLog jobId={job.id} />}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

function ConvertSummary({ job }: { job: ConvertJob }) {
  const p = job.params;
  const used = job.result?.pitch_used ?? job.detail?.pitch_used;
  const pitch =
    p.pitch_mode === "auto"
      ? `ピッチ 自動${used != null ? `（${formatPitch(used)}）` : ""}`
      : `ピッチ ${formatPitch(p.pitch)}`;
  return (
    <p className="min-w-0 text-sm text-zinc-600 dark:text-zinc-400">
      <span className="font-medium text-zinc-900 dark:text-zinc-100">{p.input_name}</span>
      {" → "}
      <span className="font-medium text-zinc-900 dark:text-zinc-100">{p.model}</span>
      ・{p.input_kind === "song" ? "曲" : "声"}・{pitch}・{formatDate(job.created_at)}
    </p>
  );
}

/** 履歴のジョブのログ（開いたときに読み込む）。失敗の理由を確かめるため。 */
function HistoryLog({ jobId }: { jobId: string }) {
  const [logs, setLogs] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  return (
    <details
      className="text-sm"
      onToggle={(event) => {
        if (!event.currentTarget.open || logs !== null) return;
        api.job(jobId, 300).then(
          (job) => setLogs(job.logs),
          (caught: unknown) => setError(caught instanceof Error ? caught.message : String(caught)),
        );
      }}
    >
      <summary className="cursor-pointer text-zinc-500 select-none hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200">
        ログを見る
      </summary>
      {error && <p className="mt-2 text-red-700 dark:text-red-300">{error}</p>}
      {logs === null && !error && <p className="mt-2 text-zinc-500 dark:text-zinc-400">読み込んでいます…</p>}
      {logs !== null && (
        <pre className="mt-2 max-h-72 overflow-auto rounded-lg bg-zinc-950 p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap text-zinc-200">
          {logs.length ? logs.join("\n") : "ログはありません。"}
        </pre>
      )}
    </details>
  );
}

function ConvertResult({
  job,
  compact = false,
  preload = "metadata",
}: {
  job: ConvertJob;
  compact?: boolean;
  preload?: "metadata" | "none";
}) {
  const result = job.result;
  const stems = result?.stems;
  const format = (result?.format ?? "wav").toUpperCase();
  return (
    <div className={compact ? "space-y-2" : "mt-4 space-y-3"}>
      {!compact && <p className="text-sm font-medium">変換した{stems ? "曲" : "音声"}</p>}
      <div className="flex flex-wrap items-center gap-3">
        <div className="min-w-60 flex-1">
          <AudioPlayer src={fileUrls.output(job.id)} label={stems ? "変換した曲" : "変換した音声"} preload={preload} />
        </div>
        <a href={fileUrls.output(job.id, true)} className={buttonClass("secondary", "sm")}>
          <DownloadIcon width={16} height={16} />
          保存（{format}）
        </a>
      </div>
      {stems && (
        <p className="flex flex-wrap items-center gap-3 text-sm">
          <span className="text-zinc-500 dark:text-zinc-400">別々に保存:</span>
          <a href={fileUrls.stem(job.id, "vocals", true)} className="inline-flex items-center gap-1 text-violet-700 hover:underline dark:text-violet-300">
            <DownloadIcon width={14} height={14} />
            変換した声だけ
          </a>
          <a href={fileUrls.stem(job.id, "instrumental", true)} className="inline-flex items-center gap-1 text-violet-700 hover:underline dark:text-violet-300">
            <DownloadIcon width={14} height={14} />
            伴奏だけ
          </a>
        </p>
      )}
      {!compact && result?.pitch_auto && result.source_pitch && (
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          声の高さ: 元の声 約{Math.round(result.source_pitch)}Hz → モデル 約{Math.round(result.target_pitch ?? 0)}Hz なので、
          ピッチを {formatPitch(result.pitch_used ?? 0)} にしました。
        </p>
      )}
      {!compact && (
        <details className="text-sm">
          <summary className="cursor-pointer text-zinc-500 select-none hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200">
            元の音声と聴き比べる
          </summary>
          <div className="mt-2">
            <AudioPlayer src={fileUrls.input(job.id)} label="元の音声" />
          </div>
        </details>
      )}
    </div>
  );
}
