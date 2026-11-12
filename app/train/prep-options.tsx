"use client";

import { useId, type ReactNode } from "react";
import type { PrepStep, SeparationQuality, SeparationStatus, SeparationStep } from "@/lib/api";
import { formatBytes, formatDuration } from "@/lib/format";
import { Code } from "@/components/ui";

/** 学習用の音声の中身。bgm: BGM・伴奏あり / speech: BGM のない話し声 / none: そのまま使う */
export type PrepKind = "bgm" | "speech" | "none";

export interface PrepChoice {
  kind: PrepKind;
  /** ハモリ（コーラス）も取り除く（BGM・伴奏ありのときだけ） */
  harmony: boolean;
  /** BGM・伴奏の除去の精度 */
  quality: SeparationQuality;
}

/** 選んだ内容から、サーバーに送る下ごしらえの手順を作る（サーバーの rvc_api/separation.py と合わせる） */
export function prepSteps({ kind, harmony }: Pick<PrepChoice, "kind" | "harmony">): PrepStep[] {
  if (kind === "bgm") return harmony ? ["vocals", "dereverb", "karaoke"] : ["vocals", "dereverb"];
  if (kind === "speech") return ["denoise"];
  return [];
}

function isSeparationStep(step: PrepStep): step is SeparationStep {
  return step !== "denoise";
}

/** まだダウンロードしていない AI モデルの合計（MB） */
export function pendingDownloadMb(
  steps: PrepStep[],
  quality: SeparationQuality,
  separation: SeparationStatus | undefined,
) {
  if (!separation) return 0;
  return steps.filter(isSeparationStep).reduce((sum, step) => {
    const info = separation.steps[step]?.[quality];
    return info ? sum + info.pending_mb : sum;
  }, 0);
}

/** 処理にかかる時間の倍率（音声の長さに対して）。Apple M5 で測った値なので、PC によって変わる */
export function separationSpeed(
  steps: PrepStep[],
  quality: SeparationQuality,
  separation: SeparationStatus | undefined,
) {
  if (!separation) return null;
  return steps.filter(isSeparationStep).reduce((sum, step) => {
    const info = separation.steps[step]?.[quality];
    return sum + (info?.speed ?? 1);
  }, 0);
}

function Option({
  name,
  checked,
  onSelect,
  disabled,
  title,
  description,
  children,
}: {
  name: string;
  checked: boolean;
  onSelect: () => void;
  disabled: boolean;
  title: string;
  description: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div
      className={`rounded-xl border p-3 text-sm transition-colors ${
        checked
          ? "border-violet-400 bg-violet-50/60 dark:border-violet-500/60 dark:bg-violet-500/10"
          : "border-zinc-200 dark:border-zinc-800"
      } ${disabled && !checked ? "opacity-60" : ""}`}
    >
      <label className={`flex items-start gap-3 ${disabled ? "" : "cursor-pointer"}`}>
        <input
          type="radio"
          name={name}
          checked={checked}
          onChange={onSelect}
          disabled={disabled}
          className="mt-0.5 size-4 shrink-0"
        />
        <span>
          <span className="font-medium">{title}</span>
          <span className="mt-0.5 block text-xs text-zinc-600 dark:text-zinc-400">
            {description}
          </span>
        </span>
      </label>
      {children}
    </div>
  );
}

/** BGM・伴奏の除去の精度を選ぶ欄（学習・変換の両方で使う） */
export function QualityChoice({
  value,
  onChange,
  disabled,
}: {
  value: SeparationQuality;
  onChange: (value: SeparationQuality) => void;
  disabled: boolean;
}) {
  const name = useId();
  const options: [SeparationQuality, string, string][] = [
    ["high", "高精度（おすすめ）", "2つの AI の結果を組み合わせます。伴奏の取り残しが少なくなります"],
    ["fast", "速さ優先", "1つの AI だけを使います。時間は約6分の1です"],
  ];
  return (
    <div role="radiogroup" aria-label="BGM・伴奏の除去の精度" className="grid gap-2 sm:grid-cols-2">
      {options.map(([option, title, description]) => (
        <label
          key={option}
          className={`flex cursor-pointer items-start gap-2 rounded-lg border p-2 ${
            value === option
              ? "border-violet-400 dark:border-violet-500/60"
              : "border-zinc-200 dark:border-zinc-800"
          }`}
        >
          <input
            type="radio"
            name={name}
            checked={value === option}
            onChange={() => onChange(option)}
            disabled={disabled}
            className="mt-0.5 size-4 shrink-0"
          />
          <span>
            <span className="font-medium">{title}</span>
            <span className="block text-xs text-zinc-600 dark:text-zinc-400">{description}</span>
          </span>
        </label>
      ))}
    </div>
  );
}

/**
 * 学習の前に音声を整える方法を選ぶ欄。
 * BGM・伴奏が入っていると、その音まで声として学習してしまい、変換した声に雑音が混ざる。
 */
export function PrepOptions({
  value,
  onChange,
  separation,
  showSetupHint,
  disabled,
}: {
  value: PrepChoice;
  onChange: (value: PrepChoice) => void;
  separation: SeparationStatus | undefined;
  /** サーバーの状態がわかっていて、BGM を取り除く準備ができていないときに案内を出す */
  showSetupHint: boolean;
  disabled: boolean;
}) {
  const name = useId();
  const harmonyId = useId();
  const installed = separation?.installed ?? false;
  const steps = prepSteps(value);
  const pendingMb = pendingDownloadMb(steps, value.quality, separation);
  const speed = separationSpeed(steps, value.quality, separation);

  return (
    <fieldset className="space-y-2">
      <legend className="mb-2 text-sm font-medium">3. 音声の下ごしらえ</legend>

      <Option
        name={name}
        checked={value.kind === "bgm"}
        onSelect={() => onChange({ ...value, kind: "bgm" })}
        disabled={disabled || !installed}
        title="BGM・伴奏が入っている"
        description="歌ってみた・カバー曲・BGM 付きの実況や配信など。AI で BGM・伴奏を取り除いて声だけにし、残響（リバーブ・エコー）も取り除いてから学習します"
      >
        {!installed && showSetupHint && (
          <p className="mt-2 pl-7 text-xs text-amber-800 dark:text-amber-300">
            この機能を使うには、<Code>npm run rvc:setup</Code>{" "}
            を実行してから、サーバーを起動し直してください。
          </p>
        )}
        {value.kind === "bgm" && installed && (
          <div className="mt-3 space-y-3 pl-7">
            <QualityChoice
              value={value.quality}
              onChange={(quality) => onChange({ ...value, quality })}
              disabled={disabled}
            />
            <label htmlFor={harmonyId} className="flex cursor-pointer items-start gap-2">
              <input
                id={harmonyId}
                type="checkbox"
                checked={value.harmony}
                onChange={(event) => onChange({ ...value, harmony: event.target.checked })}
                disabled={disabled}
                className="mt-0.5 size-4 shrink-0"
              />
              <span>
                ハモリ（コーラス）も取り除く
                <span className="block text-xs text-zinc-600 dark:text-zinc-400">
                  ハモリの多い曲で、メインの声だけを残します
                </span>
              </span>
            </label>
            <p className="text-xs text-zinc-600 dark:text-zinc-400">
              {speed != null &&
                `処理の目安: 10分の音声で約${formatDuration(speed * 600)}（Apple M5 の Mac で測った値です。NVIDIA の GPU ならもっと速くなります）。`}
              {pendingMb > 0 &&
                `初回だけ、AI モデル（約${formatBytes(pendingMb * 1_000_000)}）をダウンロードします。`}
              処理が終わると、モデル一覧で手順ごとの音声を聴き比べられます。
            </p>
          </div>
        )}
      </Option>

      <Option
        name={name}
        checked={value.kind === "speech"}
        onSelect={() => onChange({ ...value, kind: "speech" })}
        disabled={disabled}
        title="BGM のない話し声"
        description="雑談・朗読・ナレーションなど。空調の音や周りの物音などの雑音を減らしてから学習します"
      />

      <Option
        name={name}
        checked={value.kind === "none"}
        onSelect={() => onChange({ ...value, kind: "none" })}
        disabled={disabled}
        title="そのまま使う"
        description="伴奏のない歌声（アカペラ）や、静かな場所で録音したきれいな声"
      />
    </fieldset>
  );
}
