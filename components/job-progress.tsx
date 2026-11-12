"use client";

import { useEffect, useRef, useState } from "react";
import { api, isActive, type Job } from "@/lib/api";
import { formatDuration, STATUS_LABELS } from "@/lib/format";
import { useNow, usePolling } from "@/lib/hooks";
import { ConfirmButton } from "./confirm-button";
import { Alert, Badge, ProgressBar } from "./ui";

const STATUS_TONES = {
  queued: "neutral",
  running: "accent",
  done: "success",
  error: "danger",
  cancelled: "neutral",
  interrupted: "warning",
} as const;

export function StatusBadge({ status }: { status: Job["status"] }) {
  return <Badge tone={STATUS_TONES[status]}>{STATUS_LABELS[status]}</Badge>;
}

/** 学習・変換ジョブの進み具合を表示し、終わるまで自動で更新する。 */
export function JobProgress({
  initialJob,
  onFinished,
}: {
  initialJob: Job;
  onFinished?: (job: Job) => void;
}) {
  const [finished, setFinished] = useState(!isActive(initialJob.status));
  const { data, refresh } = usePolling(
    async () => {
      const job = await api.job(initialJob.id, 80);
      if (!isActive(job.status)) setFinished(true);
      return job;
    },
    finished ? null : 1500,
    initialJob.id,
  );
  const job = data ?? initialJob;
  const now = useNow();

  // 終わったときに一度だけ知らせる
  const onFinishedRef = useRef(onFinished);
  useEffect(() => {
    onFinishedRef.current = onFinished;
  });
  const reported = useRef(!isActive(initialJob.status));
  useEffect(() => {
    if (data && !isActive(data.status) && !reported.current) {
      reported.current = true;
      onFinishedRef.current?.(data);
    }
  }, [data]);

  const running = isActive(job.status);
  const elapsed =
    job.started_at && (job.finished_at || now)
      ? (job.finished_at ?? now) - job.started_at
      : null;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={job.status} />
          <span className="font-medium">
            {job.status === "queued"
              ? "前の処理が終わるのを待っています"
              : running
                ? (job.stage_label ?? "準備しています")
                : null}
          </span>
        </div>
        {running && (
          <ConfirmButton
            question="途中で止めますか？"
            confirmLabel="止める"
            onConfirm={async () => {
              await api.cancelJob(job.id);
              await refresh();
            }}
          >
            止める
          </ConfirmButton>
        )}
      </div>

      {running && (
        <ProgressBar
          value={job.status === "queued" ? 0 : job.progress}
          label={job.kind === "train" ? "学習の進み具合" : "変換の進み具合"}
        />
      )}

      <JobFacts job={job} elapsed={elapsed} />

      {job.status === "error" && job.message && (
        <Alert tone="danger" title="うまくいきませんでした">
          {job.message}
        </Alert>
      )}
      {job.status === "interrupted" && job.message && (
        <Alert tone="warning">{job.message}</Alert>
      )}
      {job.status === "done" && job.message && <Alert tone="warning">{job.message}</Alert>}

      <LogView logs={job.logs} defaultOpen={job.status === "error"} />
    </div>
  );
}

function JobFacts({ job, elapsed }: { job: Job; elapsed: number | null }) {
  const facts: string[] = [];
  if (job.kind === "train") {
    const d = job.detail;
    if (job.stage === "train" && d.total_epochs) {
      facts.push(`エポック ${d.current_epoch ?? 1} / ${d.total_epochs}`);
      if (d.batch && d.num_batches) facts.push(`バッチ ${d.batch} / ${d.num_batches}`);
      if (d.epoch_seconds) facts.push(`1エポック 約${formatDuration(d.epoch_seconds)}`);
      if (d.eta_seconds != null && isActive(job.status)) {
        facts.push(`残り 約${formatDuration(d.eta_seconds)}`);
      }
    }
    if (job.progress != null && isActive(job.status)) {
      facts.push(`全体 ${Math.round(job.progress * 100)}%`);
    }
  } else if (job.params.input_duration) {
    facts.push(`音声の長さ ${formatDuration(job.params.input_duration)}`);
  }
  if (elapsed != null && elapsed >= 0) {
    facts.push(`${isActive(job.status) ? "経過" : "かかった時間"} ${formatDuration(elapsed)}`);
  }
  if (facts.length === 0) return null;
  return (
    <p className="flex flex-wrap gap-x-4 gap-y-1 text-sm tabular-nums text-zinc-600 dark:text-zinc-400">
      {facts.map((fact) => (
        <span key={fact}>{fact}</span>
      ))}
    </p>
  );
}

function LogView({ logs, defaultOpen }: { logs: string[]; defaultOpen: boolean }) {
  const preRef = useRef<HTMLPreElement>(null);
  const last = logs.at(-1);

  // 新しいログが来たら一番下までスクロールする
  useEffect(() => {
    const pre = preRef.current;
    if (pre) pre.scrollTop = pre.scrollHeight;
  }, [logs.length, last]);

  if (logs.length === 0) return null;
  return (
    <details open={defaultOpen} className="group text-sm">
      <summary className="cursor-pointer select-none text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200">
        ログを見る
      </summary>
      <pre
        ref={preRef}
        className="mt-2 max-h-64 overflow-auto rounded-lg bg-zinc-950 p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap text-zinc-200"
      >
        {logs.join("\n")}
      </pre>
    </details>
  );
}
