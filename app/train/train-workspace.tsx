"use client";

import Link from "next/link";
import { api, fileUrls, isActive, type TrainJob } from "@/lib/api";
import { formatDate, formatDuration, formatPrep } from "@/lib/format";
import { usePolling } from "@/lib/hooks";
import { AudioPlayer } from "@/components/audio-player";
import { DatasetPreview } from "@/components/dataset-preview";
import { JobProgress, StatusBadge } from "@/components/job-progress";
import { LossChart } from "@/components/loss-chart";
import { Card, EmptyState, SectionTitle } from "@/components/ui";
import { TrainForm } from "./train-form";

/** 学習用の音声の下ごしらえが終わったあとの段階 */
const AFTER_PREP_STAGES = ["pretrained", "preprocess", "extract", "train", "evaluate", "index"];

export function TrainWorkspace() {
  const { data, refresh } = usePolling(
    () => api.jobs("train", 30) as Promise<TrainJob[]>,
    3000,
  );
  const jobs = data ?? [];
  const active = jobs.filter((job) => isActive(job.status));
  const past = jobs.filter((job) => !isActive(job.status));

  return (
    <div className="space-y-6">
      {active.length > 0 && (
        <Card aria-labelledby="training-now">
          <SectionTitle>
            <span id="training-now">学習の状況</span>
          </SectionTitle>
          <div className="divide-y divide-zinc-200 dark:divide-zinc-800">
            {active.map((job) => (
              <div key={job.id} className="space-y-3 py-4 first:pt-0 last:pb-0">
                <TrainJobTitle job={job} />
                <JobProgress initialJob={job} onFinished={() => void refresh()} />
                <EvaluationSummary job={job} />
                {job.params.prep?.some((step) => step !== "denoise") &&
                  AFTER_PREP_STAGES.includes(job.stage ?? "") && (
                    <DatasetPreview
                      model={job.params.name}
                      title="BGM・伴奏を取り除いた音声を聴く（うまく取れていなければ、止めて設定を変えられます）"
                    />
                  )}
              </div>
            ))}
          </div>
        </Card>
      )}

      <TrainForm onStarted={() => void refresh()} />

      <Card aria-labelledby="training-history">
        <SectionTitle>
          <span id="training-history">これまでの学習</span>
        </SectionTitle>
        {past.length === 0 ? (
          <EmptyState>まだ学習したことがありません。</EmptyState>
        ) : (
          <ul className="divide-y divide-zinc-200 dark:divide-zinc-800">
            {past.map((job) => (
              <TrainHistoryItem key={job.id} job={job} />
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

/** 学習中の評価（保存したモデルごとの検証用の誤差と、試し聴き）。 */
function EvaluationSummary({ job }: { job: TrainJob }) {
  const d = job.detail;
  const evaluations = d.evaluations ?? [];
  const scored = evaluations.filter((item) => item.val_loss != null);
  const latest = evaluations.at(-1);
  if (!latest) {
    if (job.stage !== "train") return null;
    return (
      <p className="text-sm text-zinc-500 dark:text-zinc-400">
        {d.early_stop
          ? "モデルが保存されるたびに品質を測り、学習しすぎになる前に自動で止めます。"
          : "モデルが保存されるたびに、試し聴き用の音声を作ります。"}
      </p>
    );
  }
  const best = scored.find((item) => item.epoch === d.best_epoch);
  return (
    <div className="space-y-2 rounded-xl border border-zinc-200 p-3 text-sm dark:border-zinc-800">
      <p className="flex flex-wrap gap-x-4 gap-y-1 tabular-nums text-zinc-600 dark:text-zinc-400">
        <span>
          最新の評価: {latest.epoch}エポック
          {latest.val_loss != null && `・誤差 ${latest.val_loss.toFixed(3)}`}
        </span>
        {best && best.val_loss != null && (
          <span className="font-medium text-zinc-900 dark:text-zinc-100">
            いちばん良い: {best.epoch}エポック（誤差 {best.val_loss.toFixed(3)}）
          </span>
        )}
        {d.early_stop && <span>学習しすぎたら自動で止めます</span>}
      </p>
      {scored.length > 1 && (
        <LossChart
          points={scored.map((item) => [item.epoch, item.val_loss as number])}
          label={`${job.params.name} の検証用の誤差（エポックごと）`}
          highlight={d.best_epoch}
          highlightLabel="いちばん良い"
        />
      )}
      {latest.sample && (
        <div className="space-y-1">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">{latest.epoch}エポックの試し聴き</p>
          <AudioPlayer
            key={latest.epoch}
            src={fileUrls.sample(job.params.name, latest.epoch)}
            label={`${latest.epoch}エポックの試し聴き`}
          />
        </div>
      )}
    </div>
  );
}

/** 「音声 3個（12分）」「動画 2本から（25分・声 18分）・BGM除去・残響除去」のような学習データの説明。 */
function describeData(job: TrainJob) {
  const p = job.params;
  const lengths = [
    p.dataset_duration ? formatDuration(p.dataset_duration) : null,
    p.voiced_duration != null ? `声 ${formatDuration(p.voiced_duration)}` : null,
  ].filter(Boolean);
  const duration = lengths.length ? `（${lengths.join("・")}）` : "";
  const source = p.sources?.length
    ? `動画 ${p.sources.length}本から${duration}`
    : `音声 ${p.dataset_files}個${duration}`;
  // 以前の版の記録（cleanup）にも対応する
  const prep = formatPrep(p.prep ?? (p.cleanup ? ["denoise"] : []));
  return prep ? `${source}・${prep}` : source;
}

function TrainJobTitle({ job }: { job: TrainJob }) {
  const p = job.params;
  return (
    <p className="text-sm text-zinc-600 dark:text-zinc-400">
      <span className="mr-2 text-base font-semibold text-zinc-900 dark:text-zinc-100">
        {p.name}
      </span>
      {p.resume ? "続きから学習" : "新しく学習"}・{p.total_epochs}エポック・
      {p.sample_rate / 1000}kHz・{describeData(job)}
    </p>
  );
}

function TrainHistoryItem({ job }: { job: TrainJob }) {
  const p = job.params;
  const took =
    job.started_at && job.finished_at ? job.finished_at - job.started_at : null;
  return (
    <li className="py-3 first:pt-0 last:pb-0">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-medium">{p.name}</span>
        <StatusBadge status={job.status} />
        <span className="text-sm text-zinc-500 dark:text-zinc-400">
          {formatDate(job.created_at)}・{p.resume ? "続きから" : "新規"}・{p.total_epochs}エポック
          {!p.resume && `・${describeData(job)}`}
          {took != null && `・${formatDuration(took)}`}
          {job.result?.early_stopped_at != null && `・${job.result.early_stopped_at}エポックで自動停止`}
          {job.result?.recommended_epoch != null && `・おすすめ ${job.result.recommended_epoch}エポック`}
        </span>
        {job.status === "done" && (
          <Link
            href={`/convert?model=${encodeURIComponent(p.name)}`}
            className="ml-auto text-sm font-medium text-violet-700 hover:underline dark:text-violet-300"
          >
            この声で変換する
          </Link>
        )}
      </div>
      {job.message && job.status !== "done" && (
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">{job.message}</p>
      )}
    </li>
  );
}
