"use client";

import Link from "next/link";
import { useState } from "react";
import { api, fileUrls, type QualityInfo, type VoiceModel } from "@/lib/api";
import { AudioPlayer } from "@/components/audio-player";
import { ConfirmButton } from "@/components/confirm-button";
import { LossChart } from "@/components/loss-chart";
import { Alert, Badge, Button } from "@/components/ui";

/** 一度に並べるエポックの数（多すぎると見づらいため） */
const VISIBLE_ROWS = 12;

/**
 * 学習したモデルの品質を確かめる欄。
 * - 検証用の誤差（学習に使っていない音声をどれだけ再現できるか）のグラフ
 * - 途中で保存したモデルごとの試し聴き（決まった話し声をそのモデルの声にしたもの）
 */
export function ModelQuality({ model, onChanged }: { model: VoiceModel; onChanged: () => void }) {
  const [info, setInfo] = useState<QualityInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);
  const busy = model.active_job_id !== null;

  async function load() {
    setError(null);
    try {
      setInfo(await api.quality(model.name));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  const evaluated = info?.evaluations.filter((item) => item.available) ?? [];
  const scored = evaluated.filter((item) => item.val_loss != null);
  const rows = [...evaluated].reverse();
  const visible = showAll ? rows : rows.slice(0, VISIBLE_ROWS);
  const trainingCurve = info?.curve.mel ?? [];

  return (
    <details
      className="text-sm"
      onToggle={(event) => {
        if (event.currentTarget.open && info === null) void load();
      }}
    >
      <summary className="cursor-pointer text-zinc-500 select-none hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200">
        品質を確かめる（エポックごとの試し聴き・グラフ）
      </summary>
      <div className="mt-3 space-y-4">
        {error && <p className="text-red-700 dark:text-red-300">{error}</p>}
        {info === null && !error && <p className="text-zinc-500 dark:text-zinc-400">読み込んでいます…</p>}
        {info && evaluated.length === 0 && (
          <p className="text-zinc-500 dark:text-zinc-400">
            まだ評価したモデルがありません（学習の途中でモデルが保存されると、ここに試し聴き用の音声が並びます）。
          </p>
        )}

        {info && scored.length > 1 && (
          <div className="space-y-1">
            <p className="font-medium">検証用の誤差（小さいほど良い）</p>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              学習に使っていない音声（{info.validation_clips ?? 0}個）を、そのエポックのモデルでどれだけ再現できたかです。
              下がりきって上がり始めたあたりからが「学習しすぎ」です。
            </p>
            <LossChart
              points={scored.map((item) => [item.epoch, item.val_loss as number])}
              label={`${model.name} の検証用の誤差（エポックごと）`}
              highlight={info.recommended_epoch}
            />
          </div>
        )}
        {info && info.validation_clips != null && info.validation_clips > 0 && info.validation_clips < 10 && scored.length > 1 && (
          <Alert tone="neutral">
            学習用の音声が短く、検証用の音声が {info.validation_clips}個（約{info.validation_clips * 3}秒）しかないため、
            誤差は上下しやすく「おすすめ」は目安です。気になるときは、前後のエポックも試し聴きで比べてください。
          </Alert>
        )}
        {info && info.validation_clips === 0 && (
          <Alert tone="neutral">
            学習用の音声が短かったため、検証用の音声を取り分けていません。試し聴きで、自然に聞こえるエポックを選んでください。
          </Alert>
        )}
        {info?.early_stopped_at != null && (
          <Alert tone="success">
            学習しすぎになる前に、{info.early_stopped_at}エポックで自動的に止めました。
          </Alert>
        )}

        {rows.length > 0 && (
          <div className="space-y-2">
            <p className="font-medium">エポックごとの試し聴き</p>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              同じ話し声を、それぞれのエポックのモデルの声にしたものです（声の高さはこのモデルに合わせています）。
            </p>
            <ul className="divide-y divide-zinc-200 rounded-xl border border-zinc-200 dark:divide-zinc-800 dark:border-zinc-800">
              {visible.map((item) => (
                <li key={item.epoch} className="space-y-2 px-3 py-2">
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                    <span className="font-medium tabular-nums">{item.epoch}エポック</span>
                    {item.epoch === info?.recommended_epoch && <Badge tone="success">おすすめ</Badge>}
                    {item.weight === model.latest_weight && <Badge tone="neutral">最新</Badge>}
                    <span className="text-xs tabular-nums text-zinc-500 dark:text-zinc-400">
                      {item.val_loss != null ? `誤差 ${item.val_loss.toFixed(3)}` : "誤差なし"}
                    </span>
                    <span className="ml-auto flex flex-wrap items-center gap-2">
                      {item.has_sample && (
                        <Button
                          size="sm"
                          variant={playing === item.epoch ? "primary" : "secondary"}
                          onClick={() => setPlaying(playing === item.epoch ? null : item.epoch)}
                          aria-expanded={playing === item.epoch}
                        >
                          試し聴き
                        </Button>
                      )}
                      {item.weight && (
                        <Link
                          href={`/convert?model=${encodeURIComponent(model.name)}&weight=${encodeURIComponent(item.weight)}`}
                          className="text-sm font-medium text-violet-700 hover:underline dark:text-violet-300"
                        >
                          これで変換
                        </Link>
                      )}
                      {item.weight && !busy && rows.length > 1 && (
                        <ConfirmButton
                          question="このエポックのモデルを削除しますか？"
                          confirmLabel="削除する"
                          onConfirm={async () => {
                            setError(null);
                            try {
                              await api.deleteWeight(model.name, item.weight as string);
                              await load();
                              onChanged();
                            } catch (caught) {
                              setError(caught instanceof Error ? caught.message : String(caught));
                            }
                          }}
                        >
                          削除
                        </ConfirmButton>
                      )}
                    </span>
                  </div>
                  {playing === item.epoch && (
                    <AudioPlayer src={fileUrls.sample(model.name, item.epoch)} label={`${item.epoch}エポックの試し聴き`} />
                  )}
                </li>
              ))}
            </ul>
            {rows.length > VISIBLE_ROWS && (
              <Button size="sm" variant="ghost" onClick={() => setShowAll(!showAll)}>
                {showAll ? "少なく表示する" : `ほかの ${rows.length - VISIBLE_ROWS}個も表示する`}
              </Button>
            )}
          </div>
        )}

        {trainingCurve.length > 1 && (
          <details>
            <summary className="cursor-pointer text-xs text-zinc-500 select-none dark:text-zinc-400">
              参考: 学習用の音声での誤差（学習の記録）
            </summary>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
              学習に使っている音声での誤差です。学習しすぎても下がり続けるので、良し悪しは上の「検証用の誤差」で判断してください。
            </p>
            <LossChart
              points={trainingCurve}
              label={`${model.name} の学習用の音声での誤差（エポックごと）`}
              format={(value) => value.toFixed(1)}
            />
          </details>
        )}
      </div>
    </details>
  );
}
