"use client";

import { useState } from "react";
import { api, type ModelStorage, type VoiceModel } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { ConfirmButton } from "@/components/confirm-button";
import { Alert } from "@/components/ui";

/** モデルが使っている保存容量の内訳と、いらないファイルの整理。 */
export function ModelStorageDetails({ model, onChanged }: { model: VoiceModel; onChanged: () => void }) {
  const [storage, setStorage] = useState<ModelStorage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [freed, setFreed] = useState<number | null>(null);
  const busy = model.active_job_id !== null;

  async function load() {
    setError(null);
    try {
      setStorage((await api.model(model.name)).storage ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function clean(options: { weights?: boolean; checkpoints?: boolean; stages?: boolean }) {
    setError(null);
    try {
      const result = await api.cleanup(model.name, options);
      setStorage(result.storage);
      setFreed(result.freed);
      onChanged();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  const total = storage
    ? storage.weights + storage.checkpoints + storage.training_files + storage.index + storage.samples + storage.dataset + storage.stages
    : 0;
  const rows: [string, number][] = storage
    ? [
        ["変換に使うモデルファイル", storage.weights],
        ["続きから学習するためのファイル", storage.checkpoints + storage.training_files],
        ["インデックス", storage.index],
        ["学習用の音声", storage.dataset],
        ["下ごしらえの途中経過（聴き比べ用）", storage.stages],
        ["試し聴き用の音声・評価結果", storage.samples],
      ]
    : [];

  return (
    <details
      className="text-sm"
      onToggle={(event) => {
        if (event.currentTarget.open && storage === null) void load();
      }}
    >
      <summary className="cursor-pointer text-zinc-500 select-none hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200">
        保存容量を見る・整理する
      </summary>
      <div className="mt-3 space-y-3">
        {error && <Alert tone="danger">{error}</Alert>}
        {freed !== null && <Alert tone="success">{formatBytes(freed)} を空けました。</Alert>}
        {storage === null && !error && <p className="text-zinc-500 dark:text-zinc-400">読み込んでいます…</p>}
        {storage && (
          <>
            <table className="w-full text-left text-sm">
              <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                {rows
                  .filter(([, size]) => size > 0)
                  .map(([label, size]) => (
                    <tr key={label}>
                      <th scope="row" className="py-1.5 font-normal text-zinc-600 dark:text-zinc-400">{label}</th>
                      <td className="py-1.5 text-right tabular-nums">{formatBytes(size)}</td>
                    </tr>
                  ))}
                <tr>
                  <th scope="row" className="py-1.5 font-medium">合計</th>
                  <td className="py-1.5 text-right font-medium tabular-nums">{formatBytes(total)}</td>
                </tr>
              </tbody>
            </table>
            {!busy && (
              <div className="space-y-2">
                {storage.extra_weight_count > 0 && (
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-zinc-600 dark:text-zinc-400">
                      おすすめと最新以外のエポック（{storage.extra_weight_count}個・{formatBytes(storage.extra_weights)}）
                    </span>
                    <ConfirmButton
                      question="おすすめと最新以外のエポックを削除しますか？"
                      confirmLabel="削除する"
                      onConfirm={() => clean({ weights: true })}
                    >
                      削除して空ける
                    </ConfirmButton>
                  </div>
                )}
                {storage.checkpoints + storage.training_files > 0 && (
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-zinc-600 dark:text-zinc-400">
                      学習を完了にする（続きから学習できなくなります・{formatBytes(storage.checkpoints + storage.training_files)}）
                    </span>
                    <ConfirmButton
                      question="続きから学習するためのファイルを削除しますか？"
                      confirmLabel="完了にする"
                      onConfirm={() => clean({ checkpoints: true })}
                    >
                      完了にする
                    </ConfirmButton>
                  </div>
                )}
                {storage.stages > 0 && (
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-zinc-600 dark:text-zinc-400">
                      下ごしらえの途中経過の音声（{formatBytes(storage.stages)}）
                    </span>
                    <ConfirmButton
                      question="聴き比べ用の音声を削除しますか？"
                      confirmLabel="削除する"
                      onConfirm={() => clean({ stages: true })}
                    >
                      削除して空ける
                    </ConfirmButton>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </details>
  );
}
