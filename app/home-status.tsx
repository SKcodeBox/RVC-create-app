"use client";

import Link from "next/link";
import { api } from "@/lib/api";
import { formatDate, formatSampleRate } from "@/lib/format";
import { usePolling } from "@/lib/hooks";
import { useServerStatus } from "@/components/server-status";
import { Badge, Card, EmptyState, ProgressBar, SectionTitle } from "@/components/ui";

/** ホーム画面の「いま動いている処理」と「最近のモデル」。 */
export function HomeStatus() {
  const { state, system } = useServerStatus();
  const { data: models } = usePolling(api.models, state === "offline" ? null : 10000, state);
  const activeJobs = system?.active_jobs ?? [];
  const recent = (models ?? []).slice(0, 4);

  if (state === "offline" || state === "checking") return null;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card aria-labelledby="active-jobs">
        <SectionTitle>
          <span id="active-jobs">実行中の処理</span>
        </SectionTitle>
        {activeJobs.length === 0 ? (
          <EmptyState>いま動いている処理はありません。</EmptyState>
        ) : (
          <ul className="space-y-4">
            {activeJobs.map((job) => (
              <li key={job.id}>
                <Link
                  href={job.kind === "train" ? "/train" : "/convert"}
                  className="block space-y-2 rounded-xl p-2 -m-2 hover:bg-zinc-50 dark:hover:bg-zinc-800/50"
                >
                  <span className="flex flex-wrap items-center justify-between gap-2 text-sm">
                    <span className="font-medium">
                      {job.kind === "train"
                        ? `「${job.params.name}」を学習`
                        : `「${job.params.input_name}」を変換`}
                    </span>
                    <span className="text-zinc-500 dark:text-zinc-400">
                      {job.status === "queued" ? "順番待ち" : (job.stage_label ?? "準備中")}
                    </span>
                  </span>
                  <ProgressBar
                    value={job.status === "queued" ? 0 : job.progress}
                    label="進み具合"
                  />
                </Link>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card aria-labelledby="recent-models">
        <SectionTitle
          action={
            <Link
              href="/models"
              className="text-sm font-medium text-violet-700 hover:underline dark:text-violet-300"
            >
              すべて見る
            </Link>
          }
        >
          <span id="recent-models">最近のモデル</span>
        </SectionTitle>
        {models === undefined ? (
          <EmptyState>読み込み中…</EmptyState>
        ) : recent.length === 0 ? (
          <EmptyState>まだモデルがありません。「声のモデルを作る」から始めましょう。</EmptyState>
        ) : (
          <ul className="divide-y divide-zinc-200 dark:divide-zinc-800">
            {recent.map((model) => (
              <li key={model.name} className="flex flex-wrap items-center gap-2 py-2.5 first:pt-0 last:pb-0">
                <span className="font-medium break-all">{model.name}</span>
                {model.active_job_id ? (
                  <Badge tone="accent">学習中</Badge>
                ) : model.weights.length > 0 ? (
                  <Badge tone="success">使える</Badge>
                ) : (
                  <Badge tone="warning">未完成</Badge>
                )}
                <span className="text-xs text-zinc-500 dark:text-zinc-400">
                  {formatSampleRate(model.sample_rate)}・{formatDate(model.created_at)}
                </span>
                {model.weights.length > 0 && !model.active_job_id && (
                  <Link
                    href={`/convert?model=${encodeURIComponent(model.name)}`}
                    className="ml-auto text-sm font-medium text-violet-700 hover:underline dark:text-violet-300"
                  >
                    変換する
                  </Link>
                )}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
