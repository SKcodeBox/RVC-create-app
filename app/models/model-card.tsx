"use client";

import { useRouter } from "next/navigation";
import { useId, useState } from "react";
import { api, EMBEDDER_LABELS, fileUrls, type VoiceModel, type VoiceType } from "@/lib/api";
import {
  formatBytes,
  formatDate,
  formatDuration,
  formatHz,
  formatPrep,
  formatPretrained,
  formatSampleRate,
} from "@/lib/format";
import { ConfirmButton } from "@/components/confirm-button";
import { DatasetPreview } from "@/components/dataset-preview";
import { DownloadIcon } from "@/components/icons";
import { Alert, Badge, Button, ButtonLink, Card, inputClass } from "@/components/ui";
import { ModelQuality } from "./model-quality";
import { ModelStorageDetails } from "./model-storage";

function ModelStatus({ model }: { model: VoiceModel }) {
  if (model.active_job_status === "running") return <Badge tone="accent">学習中</Badge>;
  if (model.active_job_status === "queued") return <Badge tone="neutral">学習待ち</Badge>;
  if (model.weights.length > 0 && model.finished === false) return <Badge tone="warning">途中で止まった</Badge>;
  if (model.weights.length > 0) return <Badge tone="success">使える</Badge>;
  return <Badge tone="warning">未完成</Badge>;
}

export function ModelCard({
  model,
  onChanged,
}: {
  model: VoiceModel;
  onChanged: () => void;
}) {
  const [resuming, setResuming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busy = model.active_job_id !== null;
  const usable = model.weights.length > 0;

  const facts = [
    model.source === "imported" ? "取り込み" : `${model.trained_epochs}エポック学習`,
    model.recommended_epoch != null ? `おすすめ ${model.recommended_epoch}エポック` : null,
    model.early_stopped_at != null ? "学習しすぎる前に自動停止" : null,
    formatSampleRate(model.sample_rate),
    model.dataset_duration ? `学習用の音声 ${formatDuration(model.dataset_duration)}` : null,
    formatPrep(model.prep) || null,
    model.source === "trained" ? formatPretrained(model.pretrained ?? "standard") : null,
    model.embedder && model.embedder !== "contentvec" ? EMBEDDER_LABELS[model.embedder] : null,
    model.target_pitch ? `声の高さ ${formatHz(model.target_pitch)}` : null,
    model.index_file ? "インデックスあり" : "インデックスなし",
    formatDate(model.created_at),
  ].filter((fact): fact is string => Boolean(fact) && fact !== "—");

  return (
    <Card className="flex flex-col gap-4">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-lg font-semibold break-all">{model.name}</h3>
          <ModelStatus model={model} />
        </div>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">{facts.join("・")}</p>
        {!usable && !busy && (
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            学習が終わる前に止まったモデルです。
            {model.can_resume ? "「続きから学習」で再開できます。" : "削除して作り直してください。"}
          </p>
        )}
        {usable && !busy && model.finished === false && (
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            学習が途中で止まりました（キャンセル・中断など）。{model.trained_epochs}エポックまでのモデルは変換に使えます。
            {model.can_resume ? "「続きから学習」で再開できます。" : ""}
          </p>
        )}
        {model.checkpoint_broken && !busy && (
          <p className="mt-2 text-sm text-amber-700 dark:text-amber-300">
            再開用のファイルが壊れているため、続きから学習はできません（保存の途中で止まった可能性があります）。
            {usable ? "保存済みのモデルはそのまま変換に使えます。" : ""}
          </p>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {usable && (
          <ButtonLink href={`/convert?model=${encodeURIComponent(model.name)}`} size="sm">
            この声で変換
          </ButtonLink>
        )}
        {busy && (
          <ButtonLink href="/train" size="sm" variant="secondary">
            学習の状況を見る
          </ButtonLink>
        )}
        {model.can_resume && !busy && !resuming && (
          <Button size="sm" variant="secondary" onClick={() => setResuming(true)}>
            続きから学習
          </Button>
        )}
        {!busy && (
          <ConfirmButton
            question="モデルと学習用の音声を削除しますか？"
            confirmLabel="削除する"
            onConfirm={async () => {
              setError(null);
              try {
                await api.deleteModel(model.name);
                onChanged();
              } catch (caught) {
                setError(caught instanceof Error ? caught.message : String(caught));
              }
            }}
          >
            削除
          </ConfirmButton>
        )}
      </div>

      {resuming && (
        <ResumeForm model={model} onCancel={() => setResuming(false)} onError={setError} />
      )}
      {error && <Alert tone="danger">{error}</Alert>}

      {model.source === "imported" && <VoiceTypeSetting model={model} onChanged={onChanged} />}

      {model.source === "trained" && usable && <ModelQuality model={model} onChanged={onChanged} />}

      {(model.weights.length > 0 || model.index_file) && (
        <details className="text-sm">
          <summary className="cursor-pointer text-zinc-500 select-none hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200">
            ファイルを保存する（{model.weights.length + (model.index_file ? 1 : 0)}個）
          </summary>
          <ul className="mt-2 space-y-1">
            {[...model.weights].reverse().map((weight) => (
              <FileLink
                key={weight.file}
                model={model.name}
                file={weight.file}
                label={
                  weight.epoch != null
                    ? `モデル（${weight.epoch}エポック${weight.file === model.recommended_weight ? "・おすすめ" : ""}${weight.file === model.latest_weight ? "・最新" : ""}）`
                    : "モデル"
                }
                size={weight.size}
              />
            ))}
            {model.index_file && (
              <FileLink model={model.name} file={model.index_file} label="インデックス" />
            )}
          </ul>
          <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
            .pth と .index は、ほかの RVC 対応アプリでも使えます。
          </p>
        </details>
      )}

      {model.source === "trained" && (
        <DatasetPreview model={model.name} title="学習に使った音声を聴く（下ごしらえの前後を聴き比べられます）" />
      )}

      <ModelStorageDetails model={model} onChanged={onChanged} />
    </Card>
  );
}

/** 取り込んだモデルの声の高さ（声の高さを自動で合わせるのに使う）。 */
function VoiceTypeSetting({ model, onChanged }: { model: VoiceModel; onChanged: () => void }) {
  const id = useId();
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <label htmlFor={id} className="text-zinc-600 dark:text-zinc-400">
        このモデルの声の高さ
      </label>
      <select
        id={id}
        className={`${inputClass} w-auto`}
        value={model.voice_type ?? ""}
        disabled={saving}
        onChange={async (event) => {
          setSaving(true);
          setError(null);
          try {
            await api.setVoiceType(model.name, (event.target.value || null) as VoiceType | null);
            onChanged();
          } catch (caught) {
            setError(caught instanceof Error ? caught.message : String(caught));
          } finally {
            setSaving(false);
          }
        }}
      >
        <option value="">わからない（声の高さの自動調整は使えません）</option>
        <option value="low">低め（男性の声など）</option>
        <option value="high">高め（女性の声など）</option>
      </select>
      {error && <span className="text-red-700 dark:text-red-300">{error}</span>}
    </div>
  );
}

function FileLink({
  model,
  file,
  label,
  size,
}: {
  model: string;
  file: string;
  label: string;
  size?: number;
}) {
  return (
    <li>
      <a
        href={fileUrls.modelFile(model, file)}
        className="inline-flex items-center gap-1.5 text-violet-700 hover:underline dark:text-violet-300"
      >
        <DownloadIcon width={14} height={14} />
        {label}
      </a>
      <span className="ml-2 text-xs text-zinc-500 dark:text-zinc-400">
        {file}
        {size != null && `・${formatBytes(size)}`}
      </span>
    </li>
  );
}

function ResumeForm({
  model,
  onCancel,
  onError,
}: {
  model: VoiceModel;
  onCancel: () => void;
  onError: (message: string | null) => void;
}) {
  const id = useId();
  const router = useRouter();
  const current = model.trained_epochs;
  const [total, setTotal] = useState(current + 100);
  const [sending, setSending] = useState(false);

  return (
    <form
      className="space-y-3 rounded-xl border border-zinc-200 p-3 dark:border-zinc-800"
      onSubmit={async (event) => {
        event.preventDefault();
        onError(null);
        setSending(true);
        try {
          await api.resumeTraining(model.name, total);
          router.push("/train");
        } catch (caught) {
          onError(caught instanceof Error ? caught.message : String(caught));
          setSending(false);
        }
      }}
    >
      <label htmlFor={id} className="block text-sm font-medium">
        合計で何エポックまで学習しますか？（いまは {current} エポック）
      </label>
      <div className="flex flex-wrap items-center gap-2">
        <input
          id={id}
          type="number"
          min={current + 1}
          max={10000}
          value={total}
          onChange={(event) => setTotal(Number(event.target.value) || current + 1)}
          className={`${inputClass} w-28`}
        />
        <Button type="submit" size="sm" disabled={sending || total <= current}>
          学習を再開する
        </Button>
        <Button size="sm" variant="ghost" onClick={onCancel} disabled={sending}>
          やめる
        </Button>
      </div>
    </form>
  );
}
