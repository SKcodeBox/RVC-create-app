"use client";

import { useId, useState, type FormEvent, type ReactNode } from "react";
import {
  AUDIO_ACCEPT,
  isAudioFile,
  upload,
  type ConvertJob,
  type ExportFormat,
  type HarmonyMode,
  type InputKind,
  type SeparationQuality,
  type TrainJob,
  type VoiceModel,
} from "@/lib/api";
import { formatBytes, formatDb, formatDuration, formatHz, formatPitch, formatSampleRate } from "@/lib/format";
import { AudioPlayer } from "@/components/audio-player";
import { FileDrop } from "@/components/file-drop";
import { useServerStatus } from "@/components/server-status";
import {
  Alert,
  Button,
  Field,
  Card,
  inputClass,
  ProgressBar,
  SectionTitle,
} from "@/components/ui";
import { QualityChoice, separationSpeed } from "../train/prep-options";

// ピッチの検出方法（サーバーの rvc_api/main.py の F0_METHODS と合わせる）。
// Applio 3.7.1 の「組み合わせ（hybrid）」は中身が作られておらず必ず失敗するため、選べないようにしている
const F0_METHODS = [
  { value: "rmvpe", label: "rmvpe（おすすめ・安定）" },
  { value: "fcpe", label: "fcpe（速い）" },
  { value: "crepe", label: "crepe（なめらか・時間がかかる）" },
];

const HARMONY_OPTIONS: { value: HarmonyMode; label: string; hint: string }[] = [
  { value: "together", label: "分けずに一緒に変換する（速い）", hint: "ハモリの少ない曲向け。ハモリが多いと声が揺れることがあります" },
  { value: "convert", label: "分けて、それぞれ変換する（自然）", hint: "メインの声とハモリを別々に変換して合わせます。時間は長くなります" },
  { value: "keep", label: "分けて、元の声のまま残す", hint: "ハモリは元の歌手の声のままになります" },
  { value: "remove", label: "分けて、消す", hint: "メインの声だけの曲になります" },
];

const EXPORT_FORMATS: { value: ExportFormat; label: string }[] = [
  { value: "wav", label: "WAV（劣化なし）" },
  { value: "flac", label: "FLAC（劣化なし・約半分の大きさ）" },
  { value: "mp3", label: "MP3（320kbps・小さい）" },
];

/** Applio の「子音・息の保護」の初期値。RVC では小さいほど強く保護し、0.5 で保護しない */
const DEFAULT_PROTECT = 0.33;

function pitchNote(pitch: number) {
  if (pitch === 0) return "そのまま";
  const octaves = Math.abs(pitch) / 12;
  const direction = pitch > 0 ? "上" : "下";
  return Number.isInteger(octaves)
    ? `${octaves}オクターブ${direction}`
    : `半音${Math.abs(pitch)}つ${direction}`;
}

function Slider({
  id,
  label,
  value,
  display,
  hint,
  min,
  max,
  step,
  onChange,
  disabled,
}: {
  id: string;
  label: ReactNode;
  value: number;
  display: string;
  hint?: ReactNode;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  disabled: boolean;
}) {
  return (
    <Field label={label} htmlFor={id} aside={display} hint={hint}>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full"
        disabled={disabled}
      />
    </Field>
  );
}

function Check({
  id,
  checked,
  onChange,
  title,
  hint,
  disabled,
}: {
  id: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  title: string;
  hint: string;
  disabled: boolean;
}) {
  return (
    <label htmlFor={id} className="flex cursor-pointer items-start gap-2 text-sm">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5 size-4 shrink-0"
        disabled={disabled}
      />
      <span>
        <span className="font-medium">{title}</span>
        <span className="block text-xs text-zinc-500 dark:text-zinc-400">{hint}</span>
      </span>
    </label>
  );
}

const LAST_MODEL_KEY = "rvc-studio:last-model";

/** 最後に変換に使ったモデルを覚えておく（この PC のブラウザの中だけ）。 */
function rememberModel(name: string) {
  try {
    localStorage.setItem(LAST_MODEL_KEY, name);
  } catch {
    // 保存できない環境（プライベートブラウズなど）では覚えない
  }
}

/**
 * 最初に選んでおくモデル: 最後に使ったもの → 最後まで学習が終わったものの中で一番新しいもの → 一番新しいもの。
 * （キャンセル・中断したモデルが先に選ばれないように）
 */
function defaultModel(models: VoiceModel[]): VoiceModel | undefined {
  let last: string | null = null;
  try {
    last = localStorage.getItem(LAST_MODEL_KEY);
  } catch {
    last = null;
  }
  return (
    models.find((m) => m.name === last) ??
    models.find((m) => m.finished !== false && m.active_job_id === null) ??
    models[0]
  );
}

export function ConvertForm({
  models,
  initialModel,
  initialWeight,
  onStarted,
}: {
  models: VoiceModel[];
  initialModel: string | null;
  initialWeight: string | null;
  onStarted: (job: ConvertJob) => void;
}) {
  const ids = {
    model: useId(),
    weight: useId(),
    pitch: useId(),
    method: useId(),
    index: useId(),
    protect: useId(),
    envelope: useId(),
    autotune: useId(),
    autotuneStrength: useId(),
    formant: useId(),
    quefrency: useId(),
    timbre: useId(),
    cleanInput: useId(),
    cleanOutput: useId(),
    cleanStrength: useId(),
    format: useId(),
    harmony: useId(),
    dereverb: useId(),
    reverb: useId(),
    vocalGain: useId(),
    instGain: useId(),
    pitchMode: useId(),
  };
  const { state, system } = useServerStatus();
  const [modelName, setModelName] = useState(initialModel ?? "");
  const [weight, setWeight] = useState(initialWeight ?? "");
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [notAudio, setNotAudio] = useState<string | null>(null);
  const [inputKind, setInputKind] = useState<InputKind>("voice");
  const [chosenPitchMode, setChosenPitchMode] = useState<"auto" | "manual" | null>(null);
  const [pitch, setPitch] = useState(0);
  const [f0Method, setF0Method] = useState("rmvpe");
  const [indexRate, setIndexRate] = useState(0.75);
  // 画面では「保護の強さ」（0〜1、大きいほど強い）にして、送るときに RVC の値（0.5〜0）に直す
  const [protectStrength, setProtectStrength] = useState(1 - DEFAULT_PROTECT / 0.5);
  const [volumeEnvelope, setVolumeEnvelope] = useState(1);
  const [autotune, setAutotune] = useState(false);
  const [autotuneStrength, setAutotuneStrength] = useState(1);
  const [formant, setFormant] = useState(false);
  const [quefrency, setQuefrency] = useState(1);
  const [timbre, setTimbre] = useState(1);
  const [cleanInput, setCleanInput] = useState(false);
  const [cleanOutput, setCleanOutput] = useState(false);
  const [cleanStrength, setCleanStrength] = useState(12);
  const [exportFormat, setExportFormat] = useState<ExportFormat>("wav");
  const [quality, setQuality] = useState<SeparationQuality>("high");
  const [dereverb, setDereverb] = useState(true);
  const [harmony, setHarmony] = useState<HarmonyMode>("together");
  const [reverb, setReverb] = useState(0.3);
  const [vocalGain, setVocalGain] = useState(0);
  const [instGain, setInstGain] = useState(0);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const model = models.find((m) => m.name === modelName) ?? defaultModel(models);
  const weightFile =
    model?.weights.find((w) => w.file === weight)?.file ??
    model?.recommended_weight ??
    model?.latest_weight ??
    null;
  const canAuto = Boolean(model?.target_pitch);
  const pitchMode = chosenPitchMode === "auto" && !canAuto ? "manual" : (chosenPitchMode ?? (canAuto ? "auto" : "manual"));
  const song = inputKind === "song";
  const separationReady = system?.separation?.installed ?? false;
  // 学習中のジョブ（曲の変換と並行すると、どちらも遅くなる）
  const training = system?.active_jobs.find((job): job is TrainJob => job.kind === "train" && job.status === "running") ?? null;
  const protect = Math.round((0.5 * (1 - protectStrength)) * 100) / 100;
  // 曲の変換で、声と伴奏を分けるのにかかる時間の目安（音声の長さに対する倍率）
  const songSpeed = separationSpeed(
    ["vocals", ...(dereverb ? (["dereverb"] as const) : []), ...(harmony !== "together" ? (["karaoke"] as const) : [])],
    quality,
    system?.separation,
  );
  const uploading = uploadProgress !== null;
  const canSubmit =
    state === "ready" && model && weightFile && file && !uploading && (!song || separationReady);

  function chooseFile(files: File[]) {
    const chosen = files[0];
    if (!isAudioFile(chosen)) {
      setNotAudio(chosen.name);
      return;
    }
    setNotAudio(null);
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    setFile(chosen);
    setFileUrl(URL.createObjectURL(chosen));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit || !model || !weightFile || !file) return;
    setError(null);
    const form = new FormData();
    const fields: Record<string, string | number | boolean> = {
      model: model.name,
      weight: weightFile,
      pitch,
      pitch_mode: pitchMode,
      f0_method: f0Method,
      index_rate: indexRate,
      protect,
      volume_envelope: volumeEnvelope,
      autotune,
      autotune_strength: autotuneStrength,
      formant,
      formant_quefrency: quefrency,
      formant_timbre: timbre,
      clean_input: cleanInput,
      clean_output: cleanOutput,
      clean_strength: cleanStrength,
      export_format: exportFormat,
      input_kind: inputKind,
      separation_quality: quality,
      dereverb,
      harmony,
      reverb: song && dereverb ? reverb : 0,
      vocal_gain_db: vocalGain,
      instrumental_gain_db: instGain,
    };
    for (const [key, value] of Object.entries(fields)) form.set(key, String(value));
    form.set("file", file, file.name);
    setUploadProgress(0);
    try {
      const { job } = await upload<{ job: ConvertJob }>("/api/convert", form, setUploadProgress);
      rememberModel(model.name);
      onStarted(job);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setUploadProgress(null);
    }
  }

  return (
    <Card aria-labelledby="convert-form">
      <SectionTitle>
        <span id="convert-form">変換の設定</span>
      </SectionTitle>
      <form onSubmit={handleSubmit} className="space-y-6">
        <Field
          label="1. 使うモデル"
          htmlFor={ids.model}
          hint={
            model
              ? [
                  model.source === "imported" ? "取り込んだモデル" : `${model.trained_epochs}エポック学習`,
                  model.recommended_epoch != null ? `おすすめ ${model.recommended_epoch}エポックを使います` : null,
                  formatSampleRate(model.sample_rate),
                  model.index_file ? "インデックスあり" : "インデックスなし",
                ]
                  .filter(Boolean)
                  .join("・")
              : undefined
          }
        >
          <select
            id={ids.model}
            className={inputClass}
            value={model?.name ?? ""}
            onChange={(event) => {
              setModelName(event.target.value);
              setWeight("");
              setChosenPitchMode(null);
            }}
            disabled={uploading || models.length === 0}
          >
            {models.length === 0 && <option value="">読み込み中…</option>}
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
        </Field>

        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-medium">2. 変換する音声</p>
            <div
              role="group"
              aria-label="変換する音声の種類"
              className="inline-flex rounded-lg border border-zinc-200 bg-zinc-100 p-0.5 dark:border-zinc-800 dark:bg-zinc-900"
            >
              {(
                [
                  ["voice", "声だけの音声"],
                  ["song", "曲（伴奏あり）"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={inputKind === value}
                  onClick={() => setInputKind(value)}
                  disabled={uploading}
                  className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                    inputKind === value
                      ? "bg-white font-medium text-zinc-900 shadow-sm dark:bg-zinc-700 dark:text-zinc-50"
                      : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <FileDrop
            accept={AUDIO_ACCEPT}
            disabled={uploading}
            title={file ? "別の音声を選ぶ" : "音声ファイルを選ぶ（ドラッグ＆ドロップもできます）"}
            hint={
              song
                ? "歌ってみた・カバーしたい曲など。AI で声と伴奏を分け、声だけを変換して伴奏と合わせ直します"
                : "話し声やアカペラなど、声だけの音声。伴奏が入っているときは「曲（伴奏あり）」を選んでください"
            }
            onFiles={chooseFile}
          />
          {notAudio && <Alert tone="warning">「{notAudio}」は音声ファイルではありません。</Alert>}
          {file && fileUrl && (
            <div className="space-y-1.5 rounded-xl border border-zinc-200 p-3 dark:border-zinc-800">
              <p className="truncate text-sm" title={file.name}>
                {file.name}
                <span className="ml-2 text-zinc-500 dark:text-zinc-400">{formatBytes(file.size)}</span>
              </p>
              <AudioPlayer src={fileUrl} label="変換する前の音声" />
            </div>
          )}

          {song && !separationReady && system !== null && (
            <Alert tone="warning">
              曲の変換には、BGM・伴奏を取り除く機能が必要です。npm run rvc:setup を実行してから、サーバーを起動し直してください。
            </Alert>
          )}
          {song && separationReady && training && (
            <Alert tone="warning">
              いま「{training.params.name}」を学習中です。曲を声と伴奏に分ける処理は重いため、並行すると変換に時間がかかり、
              学習も遅くなります（実測で学習が約2倍の時間に）。急がなければ、学習が終わってからの変換がおすすめです。
            </Alert>
          )}
          {song && separationReady && (
            <div className="space-y-4 rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
              <p className="text-sm font-medium">曲の変換の設定</p>
              <QualityChoice value={quality} onChange={setQuality} disabled={uploading} />
              <Field label="ハモリ（コーラス）の扱い" htmlFor={ids.harmony} hint={HARMONY_OPTIONS.find((o) => o.value === harmony)?.hint}>
                <select
                  id={ids.harmony}
                  className={inputClass}
                  value={harmony}
                  onChange={(event) => setHarmony(event.target.value as HarmonyMode)}
                  disabled={uploading}
                >
                  {HARMONY_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </Field>
              <Check
                id={ids.dereverb}
                checked={dereverb}
                onChange={setDereverb}
                title="元の声の残響（リバーブ）を取り除いてから変換する（おすすめ）"
                hint="残響が入ったまま変換すると、声がこもったり揺れたりしやすくなります"
                disabled={uploading}
              />
              {dereverb && (
                <Slider
                  id={ids.reverb}
                  label="変換した声に付け直す残響"
                  value={reverb}
                  display={reverb === 0 ? "なし" : `${Math.round(reverb * 100)}%`}
                  hint="声だけが浮いて聞こえないよう、軽い響きを付けます"
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={setReverb}
                  disabled={uploading}
                />
              )}
              <div className="grid gap-4 sm:grid-cols-2">
                <Slider
                  id={ids.vocalGain}
                  label="声の大きさ"
                  value={vocalGain}
                  display={vocalGain === 0 ? "元の曲と同じ" : formatDb(vocalGain)}
                  min={-12}
                  max={12}
                  step={0.5}
                  onChange={setVocalGain}
                  disabled={uploading}
                />
                <Slider
                  id={ids.instGain}
                  label="伴奏の大きさ"
                  value={instGain}
                  display={instGain === 0 ? "元の曲と同じ" : formatDb(instGain)}
                  min={-12}
                  max={12}
                  step={0.5}
                  onChange={setInstGain}
                  disabled={uploading}
                />
              </div>
            </div>
          )}
        </div>

        <fieldset className="space-y-3">
          <legend className="mb-2 text-sm font-medium">3. 声の高さ（ピッチ）</legend>
          <div role="radiogroup" aria-label="声の高さの決め方" className="grid gap-2 sm:grid-cols-2">
            {(
              [
                [
                  "auto",
                  "自動で合わせる（おすすめ）",
                  canAuto
                    ? `元の声の高さを調べて、モデルの声（${formatHz(model?.target_pitch)}）に近づけます${song ? "。曲はキーが変わらないよう、オクターブ単位で合わせます" : ""}`
                    : "このモデルは声の高さがわからないため使えません（モデル一覧で設定できます）",
                ],
                ["manual", "自分で決める", "上げ下げする量（半音）を指定します"],
              ] as const
            ).map(([value, title, description]) => (
              <label
                key={value}
                className={`flex items-start gap-2 rounded-lg border p-2 text-sm ${
                  pitchMode === value ? "border-violet-400 dark:border-violet-500/60" : "border-zinc-200 dark:border-zinc-800"
                } ${value === "auto" && !canAuto ? "opacity-60" : "cursor-pointer"}`}
              >
                <input
                  type="radio"
                  name={ids.pitchMode}
                  checked={pitchMode === value}
                  onChange={() => setChosenPitchMode(value)}
                  disabled={uploading || (value === "auto" && !canAuto)}
                  className="mt-0.5 size-4 shrink-0"
                />
                <span>
                  <span className="font-medium">{title}</span>
                  <span className="block text-xs text-zinc-600 dark:text-zinc-400">{description}</span>
                </span>
              </label>
            ))}
          </div>
          {pitchMode === "manual" && (
            <Field
              label="上げ下げする量"
              htmlFor={ids.pitch}
              aside={`${formatPitch(pitch)}（${pitchNote(pitch)}）`}
              hint={
                song && pitch % 12 !== 0
                  ? "オクターブ（±12）単位でないと、声だけキーが変わって伴奏と合わなくなります。"
                  : "男性の声を女性のモデルで変換するなら +12、女性の声を男性のモデルで変換するなら -12 が目安です。"
              }
            >
              <input
                id={ids.pitch}
                type="range"
                min={-24}
                max={24}
                step={1}
                value={pitch}
                onChange={(event) => setPitch(Number(event.target.value))}
                className="w-full"
                disabled={uploading}
              />
              <div className="flex gap-2">
                {[-12, 0, 12].map((preset) => (
                  <Button
                    key={preset}
                    size="sm"
                    variant={pitch === preset ? "primary" : "secondary"}
                    onClick={() => setPitch(preset)}
                    disabled={uploading}
                  >
                    {formatPitch(preset)}
                  </Button>
                ))}
              </div>
            </Field>
          )}
        </fieldset>

        <details className="rounded-xl border border-zinc-200 px-4 py-3 dark:border-zinc-800">
          <summary className="cursor-pointer text-sm font-medium select-none">
            詳しい設定（ふつうは変えなくて大丈夫です）
          </summary>
          <div className="mt-4 grid gap-5 sm:grid-cols-2">
            {model && model.weights.length > 1 && (
              <Field
                label="使うモデルファイル"
                htmlFor={ids.weight}
                hint="学習の途中で保存されたものも選べます。モデル一覧で試し聴きできます"
              >
                <select
                  id={ids.weight}
                  className={inputClass}
                  value={weightFile ?? ""}
                  onChange={(event) => setWeight(event.target.value)}
                  disabled={uploading}
                >
                  {[...model.weights].reverse().map((w) => (
                    <option key={w.file} value={w.file}>
                      {w.epoch != null ? `${w.epoch}エポック` : w.file}
                      {w.file === model.recommended_weight ? "（おすすめ）" : ""}
                      {w.file === model.latest_weight ? "（最新）" : ""}
                    </option>
                  ))}
                </select>
              </Field>
            )}
            <Field label="ピッチの検出方法" htmlFor={ids.method}>
              <select
                id={ids.method}
                className={inputClass}
                value={f0Method}
                onChange={(event) => setF0Method(event.target.value)}
                disabled={uploading}
              >
                {F0_METHODS.map((method) => (
                  <option key={method.value} value={method.value}>
                    {method.value === "crepe" && system?.platform === "mac"
                      ? "crepe（なめらか・Mac ではとても遅い）"
                      : method.label}
                  </option>
                ))}
              </select>
              {f0Method === "crepe" && system?.platform === "mac" && (
                <p className="mt-1 text-xs text-amber-700 dark:text-amber-300">
                  Mac では GPU を使えないため、とても時間がかかります（7秒の音声に約1分半、4分の曲なら1時間近く）。
                </p>
              )}
            </Field>
            <Slider
              id={ids.index}
              label="インデックスの効き具合"
              value={indexRate}
              display={indexRate.toFixed(2)}
              hint={model?.index_file ? "高いほどモデルの声に近づきますが、雑音が出ることがあります" : "このモデルにはインデックスがありません"}
              min={0}
              max={1}
              step={0.05}
              onChange={setIndexRate}
              disabled={uploading || !model?.index_file}
            />
            <Slider
              id={ids.protect}
              label="子音・息の保護の強さ"
              value={protectStrength}
              display={protectStrength === 0 ? "保護しない" : `${Math.round(protectStrength * 100)}%`}
              hint="強いほど、子音や息の音が割れにくくなります（強すぎると、子音が元の声に近く残ります）"
              min={0}
              max={1}
              step={0.02}
              onChange={setProtectStrength}
              disabled={uploading}
            />
            <Slider
              id={ids.envelope}
              label="音量の変化"
              value={volumeEnvelope}
              display={volumeEnvelope.toFixed(2)}
              hint="1 に近いほど変換後の声の音量の変化を、0 に近いほど元の音声の音量の変化を使います"
              min={0}
              max={1}
              step={0.05}
              onChange={setVolumeEnvelope}
              disabled={uploading}
            />
            <div className="space-y-2">
              <Check
                id={ids.autotune}
                checked={autotune}
                onChange={setAutotune}
                title="オートチューン"
                hint="歌声の音程を補正します"
                disabled={uploading}
              />
              {autotune && (
                <Slider
                  id={ids.autotuneStrength}
                  label="補正の強さ"
                  value={autotuneStrength}
                  display={`${Math.round(autotuneStrength * 100)}%`}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={setAutotuneStrength}
                  disabled={uploading}
                />
              )}
            </div>
            <div className="space-y-2">
              <Check
                id={ids.formant}
                checked={formant}
                onChange={setFormant}
                title="声質（フォルマント）を調整する"
                hint="男性の声を女性の声に変えるときなどに、声の太さ・響きを合わせます"
                disabled={uploading}
              />
              {formant && (
                <>
                  <Slider
                    id={ids.quefrency}
                    label="響き（ケフレンシー）"
                    value={quefrency}
                    display={quefrency.toFixed(2)}
                    min={0.5}
                    max={2}
                    step={0.05}
                    onChange={setQuefrency}
                    disabled={uploading}
                  />
                  <Slider
                    id={ids.timbre}
                    label="声色（大きいほど細く・小さいほど太く）"
                    value={timbre}
                    display={timbre.toFixed(2)}
                    min={0.5}
                    max={2}
                    step={0.05}
                    onChange={setTimbre}
                    disabled={uploading}
                  />
                </>
              )}
            </div>
            <div className="space-y-2">
              <Check
                id={ids.cleanInput}
                checked={cleanInput}
                onChange={setCleanInput}
                title="変換する前に雑音を減らす"
                hint="録音した声に空調の音などが入っているとき"
                disabled={uploading}
              />
              <Check
                id={ids.cleanOutput}
                checked={cleanOutput}
                onChange={setCleanOutput}
                title="変換した声の雑音を減らす"
                hint="変換した声にザラザラした音が残るとき"
                disabled={uploading}
              />
              {(cleanInput || cleanOutput) && (
                <Slider
                  id={ids.cleanStrength}
                  label="雑音を減らす強さ"
                  value={cleanStrength}
                  display={`${cleanStrength}dB`}
                  min={1}
                  max={30}
                  step={1}
                  onChange={setCleanStrength}
                  disabled={uploading}
                />
              )}
            </div>
            <Field label="書き出す形式" htmlFor={ids.format}>
              <select
                id={ids.format}
                className={inputClass}
                value={exportFormat}
                onChange={(event) => setExportFormat(event.target.value as ExportFormat)}
                disabled={uploading}
              >
                {EXPORT_FORMATS.filter((option) => option.value !== "mp3" || system?.ffmpeg !== false).map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </Field>
          </div>
        </details>

        {error && (
          <Alert tone="danger" title="変換を始められませんでした">
            {error}
          </Alert>
        )}
        {uploading && (
          <div className="space-y-2">
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              音声を送っています… {Math.round((uploadProgress ?? 0) * 100)}%
            </p>
            <ProgressBar value={uploadProgress} label="アップロードの進み具合" />
          </div>
        )}
        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" disabled={!canSubmit}>
            {uploading ? "送っています…" : "変換する"}
          </Button>
          {file && (
            <span className="text-sm text-zinc-500 dark:text-zinc-400">
              {song && songSpeed != null
                ? `曲の変換は、声と伴奏を分けるのに時間がかかります（4分の曲で約${formatDuration(songSpeed * 240)}・Apple M5 の Mac の場合）`
                : "音声が長いほど時間がかかります"}
            </span>
          )}
        </div>
      </form>
    </Card>
  );
}
