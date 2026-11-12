"use client";

import { useId, useState, type FormEvent } from "react";
import {
  api,
  AUDIO_ACCEPT,
  EMBEDDER_LABELS,
  isAudioFile,
  MODEL_NAME_RE,
  modelNameError,
  upload,
  type Embedder,
  type SeparationQuality,
  type TrainJob,
} from "@/lib/api";
import { formatBytes, formatDuration } from "@/lib/format";
import { usePolling } from "@/lib/hooks";
import { FileDrop } from "@/components/file-drop";
import { CloseIcon } from "@/components/icons";
import { useServerStatus } from "@/components/server-status";
import {
  Alert,
  Button,
  Card,
  Field,
  inputClass,
  ProgressBar,
  SectionTitle,
} from "@/components/ui";
import { PrepOptions, prepSteps, type PrepKind } from "./prep-options";
import { emptyVideoRow, videoRowError, VideoSources, type VideoRow } from "./video-sources";

const EPOCH_PRESETS = [100, 200, 300, 500];

type SourceMode = "files" | "videos";
const RECOMMENDED_SECONDS = 10 * 60;

function fileKey(file: File) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

/** ブラウザで音声の長さを調べる（読めない形式なら null）。 */
function probeDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    const finish = (value: number | null) => {
      URL.revokeObjectURL(url);
      resolve(value);
    };
    audio.preload = "metadata";
    audio.onloadedmetadata = () =>
      finish(Number.isFinite(audio.duration) ? audio.duration : null);
    audio.onerror = () => finish(null);
    audio.src = url;
  });
}

export function TrainForm({ onStarted }: { onStarted: (job: TrainJob) => void }) {
  const ids = {
    name: useId(),
    epochs: useId(),
    rate: useId(),
    batch: useId(),
    save: useId(),
    pretrained: useId(),
    embedder: useId(),
    earlyStop: useId(),
  };
  const { state, system } = useServerStatus();
  const [name, setName] = useState("");
  const [mode, setMode] = useState<SourceMode>("files");
  const [videoRows, setVideoRows] = useState<VideoRow[]>([emptyVideoRow(1)]);
  // 選ぶまでは、音声の用意のしかたに合わせたおすすめを使う
  const [chosenPrep, setChosenPrep] = useState<PrepKind | null>(null);
  const [harmony, setHarmony] = useState(false);
  const [quality, setQuality] = useState<SeparationQuality>("high");
  const [consent, setConsent] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [durations, setDurations] = useState<Record<string, number | null>>({});
  const [skipped, setSkipped] = useState<string[]>([]);
  const [epochs, setEpochs] = useState(300);
  const [sampleRate, setSampleRate] = useState(40000);
  // 選ぶまでは、GPU のメモリに合わせたおすすめを使う
  const [chosenBatch, setChosenBatch] = useState<number | null>(null);
  const [saveEvery, setSaveEvery] = useState(10);
  const [chosenPretrained, setChosenPretrained] = useState("standard");
  const [embedder, setEmbedder] = useState<Embedder>("contentvec");
  const [earlyStop, setEarlyStop] = useState(true);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startedName, setStartedName] = useState<string | null>(null);

  const uploading = uploadProgress !== null;
  // 同じ名前のモデルがあるかを、音声を送る前に確かめる（大きな音声を送り終えてから失敗しないように）
  const { data: models } = usePolling(api.models, 10000);
  const nameError = modelNameError(name, models ?? []);
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  const known = files.map((file) => durations[fileKey(file)]);
  const totalSeconds = known.reduce<number>((sum, value) => sum + (value ?? 0), 0);
  const measuring = known.some((value) => value === undefined);
  const filledRows = videoRows.filter((row) => row.url.trim());
  const videosReady =
    filledRows.length > 0 && videoRows.every((row) => !videoRowError(row)) && consent;
  const separation = system?.separation;
  const batchSize = chosenBatch ?? system?.recommended_batch_size ?? 4;
  const pretrains = system?.pretrains ?? [];
  const pretrainedInfo = pretrains.find((item) => item.id === chosenPretrained);
  // 選んだ事前学習モデルがそのサンプルレートに対応していなければ、標準のものを使う
  const pretrained =
    pretrainedInfo && pretrainedInfo.sample_rates.includes(sampleRate) ? chosenPretrained : "standard";
  const pretrainedReady =
    pretrains.find((item) => item.id === pretrained)?.downloaded[String(sampleRate)] ?? true;
  const separationReady = separation?.installed ?? false;
  // 動画にはたいてい BGM が入っているので、動画のときは BGM を取り除くのをおすすめにする
  const recommendedPrep: PrepKind =
    mode === "videos" ? (separationReady ? "bgm" : "speech") : "none";
  const prepKind = chosenPrep ?? recommendedPrep;
  const canSubmit =
    state === "ready" &&
    MODEL_NAME_RE.test(name) &&
    !nameError &&
    !uploading &&
    (prepKind !== "bgm" || separationReady) &&
    (mode === "files" ? files.length > 0 : videosReady);

  function addFiles(list: File[]) {
    setSkipped(list.filter((file) => !isAudioFile(file)).map((file) => file.name));
    const existing = new Set(files.map(fileKey));
    const added = list.filter((file) => isAudioFile(file) && !existing.has(fileKey(file)));
    setFiles([...files, ...added]);
    for (const file of added) {
      void probeDuration(file).then((seconds) =>
        setDurations((current) => ({ ...current, [fileKey(file)]: seconds })),
      );
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) return;
    setError(null);
    setStartedName(null);
    const form = new FormData();
    form.set("name", name);
    form.set("sample_rate", String(sampleRate));
    form.set("total_epochs", String(epochs));
    form.set("batch_size", String(batchSize));
    form.set("save_every", String(saveEvery));
    form.set("prep", prepSteps({ kind: prepKind, harmony }).join(","));
    form.set("separation_quality", quality);
    form.set("pretrained", pretrained);
    form.set("embedder", embedder);
    form.set("early_stop", String(earlyStop));
    // いま開いている方（音声ファイル／動画の URL）だけを送る
    if (mode === "files") {
      for (const file of files) form.append("files", file, file.name);
    } else {
      form.set(
        "sources",
        JSON.stringify(
          filledRows.map((row) => ({
            url: row.url.trim(),
            start: row.start.trim(),
            end: row.end.trim(),
          })),
        ),
      );
      form.set("consent", String(consent));
    }

    setUploadProgress(0);
    try {
      const { job } = await upload<{ job: TrainJob }>("/api/train", form, setUploadProgress);
      setStartedName(job.params.name);
      setName("");
      setFiles([]);
      setSkipped([]);
      setVideoRows([emptyVideoRow(1)]);
      setConsent(false);
      setChosenPrep(null);
      setHarmony(false);
      onStarted(job);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setUploadProgress(null);
    }
  }

  return (
    <Card aria-labelledby="new-model">
      <SectionTitle>
        <span id="new-model">新しいモデルを作る</span>
      </SectionTitle>
      <form onSubmit={handleSubmit} className="space-y-6">
        <Field
          label="1. モデル名"
          htmlFor={ids.name}
          hint="半角英数字・「-」・「_」が使えます（例: my-voice）"
          error={nameError}
        >
          <input
            id={ids.name}
            className={inputClass}
            value={name}
            onChange={(event) => setName(event.target.value.trim())}
            spellCheck={false}
            placeholder="my-voice"
            autoComplete="off"
            maxLength={40}
            aria-invalid={nameError ? true : undefined}
            disabled={uploading}
          />
        </Field>

        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-medium">2. 学習用の音声</p>
            <div
              role="group"
              aria-label="音声の用意のしかた"
              className="inline-flex rounded-lg border border-zinc-200 bg-zinc-100 p-0.5 dark:border-zinc-800 dark:bg-zinc-900"
            >
              {(
                [
                  ["files", "音声ファイル"],
                  ["videos", "動画の URL（YouTube など）"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={mode === value}
                  onClick={() => setMode(value)}
                  disabled={uploading}
                  className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                    mode === value
                      ? "bg-white font-medium text-zinc-900 shadow-sm dark:bg-zinc-700 dark:text-zinc-50"
                      : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {mode === "videos" ? (
            <VideoSources
              rows={videoRows}
              onRowsChange={setVideoRows}
              consent={consent}
              onConsentChange={setConsent}
              disabled={uploading}
            />
          ) : (
            <>
              <FileDrop
                accept={AUDIO_ACCEPT}
                multiple
                disabled={uploading}
                title="音声ファイルを選ぶ（ドラッグ＆ドロップもできます）"
                hint="WAV・MP3・FLAC・M4A・Opus などの音声や、MP4・MOV などの動画（音声だけを使います）。何個でもまとめて選べます"
                onFiles={addFiles}
              />
              {skipped.length > 0 && (
                <Alert tone="warning">
                  音声ではないため外したファイル: {skipped.join("、")}
                </Alert>
              )}
              {files.length > 0 && (
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800">
                  <ul className="max-h-60 divide-y divide-zinc-200 overflow-auto dark:divide-zinc-800">
                    {files.map((file) => {
                      const seconds = durations[fileKey(file)];
                      return (
                        <li key={fileKey(file)} className="flex items-center gap-3 px-3 py-2 text-sm">
                          <span className="min-w-0 flex-1 truncate" title={file.name}>
                            {file.name}
                          </span>
                          <span className="shrink-0 tabular-nums text-zinc-500 dark:text-zinc-400">
                            {seconds === undefined ? "…" : formatDuration(seconds)}・
                            {formatBytes(file.size)}
                          </span>
                          <button
                            type="button"
                            onClick={() => setFiles(files.filter((f) => f !== file))}
                            disabled={uploading}
                            className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
                            aria-label={`${file.name} を外す`}
                          >
                            <CloseIcon width={16} height={16} />
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                  <p className="border-t border-zinc-200 px-3 py-2 text-sm tabular-nums dark:border-zinc-800">
                    {files.length}個・合計 {measuring ? "計算中…" : formatDuration(totalSeconds)}・
                    {formatBytes(totalSize)}
                  </p>
                </div>
              )}
              {files.length > 0 && !measuring && totalSeconds < RECOMMENDED_SECONDS && (
                <Alert tone="warning">
                  合計が10分未満です。短くても学習はできますが、10〜30分ほどあると声の再現度が上がります。
                </Alert>
              )}
              <ul className="list-disc space-y-0.5 pl-5 text-xs text-zinc-500 dark:text-zinc-400">
                <li>
                  学習したい人の声だけが入った音声がおすすめです。BGM・伴奏は下の「音声の下ごしらえ」で取り除けますが、
                  ほかの人の声は残ります
                </li>
                <li>話し声でも歌声でも学習できます。変換したい用途に近い音声を使うと仕上がりが良くなります</li>
                <li>他の人の声を使うときは、必ず本人の許可を取ってください</li>
              </ul>
            </>
          )}
        </div>

        <PrepOptions
          value={{ kind: prepKind, harmony, quality }}
          onChange={(value) => {
            setChosenPrep(value.kind);
            setHarmony(value.harmony);
            setQuality(value.quality);
          }}
          separation={separation}
          showSetupHint={system !== null}
          disabled={uploading}
        />

        <div className="space-y-4">
          <p className="text-sm font-medium">4. 学習の設定</p>
          <Field
            label="エポック数（学習を繰り返す回数の上限）"
            htmlFor={ids.epochs}
            hint={
              earlyStop
                ? "多めにしておけば大丈夫です。学習しすぎになる前に自動で止まり、いちばん良いところのモデルを選びます。"
                : "多いほど本人の声に近づきますが、多すぎると学習しすぎで声が崩れます。迷ったら 200 にしてください。"
            }
          >
            <div className="flex flex-wrap items-center gap-2">
              <input
                id={ids.epochs}
                type="number"
                min={1}
                max={10000}
                className={`${inputClass} w-28`}
                value={epochs}
                onChange={(event) => setEpochs(Number(event.target.value) || 1)}
                disabled={uploading}
              />
              {EPOCH_PRESETS.map((preset) => (
                <Button
                  key={preset}
                  size="sm"
                  variant={preset === epochs ? "primary" : "secondary"}
                  onClick={() => setEpochs(preset)}
                  disabled={uploading}
                >
                  {preset}
                </Button>
              ))}
            </div>
          </Field>

          <label htmlFor={ids.earlyStop} className="flex cursor-pointer items-start gap-2 text-sm">
            <input
              id={ids.earlyStop}
              type="checkbox"
              checked={earlyStop}
              onChange={(event) => setEarlyStop(event.target.checked)}
              disabled={uploading}
              className="mt-0.5 size-4 shrink-0"
            />
            <span>
              <span className="font-medium">学習しすぎたら自動で止める（おすすめ）</span>
              <span className="block text-xs text-zinc-500 dark:text-zinc-400">
                音声の一部（約4%）を学習に使わずに取り分け、保存のたびに「初めて聞く音声をどれだけ再現できるか」を測ります。
                良くならなくなったら止めます。止めない場合も、いちばん良かったエポックが変換に使われます
              </span>
            </span>
          </label>

          <Field
            label="事前学習モデル（学習の出発点）"
            htmlFor={ids.pretrained}
            hint={
              <>
                {pretrains.find((item) => item.id === pretrained)?.description}
                {!pretrainedReady && "。初回だけダウンロードします（約1.3GB）"}
              </>
            }
          >
            <select
              id={ids.pretrained}
              className={inputClass}
              value={pretrained}
              onChange={(event) => setChosenPretrained(event.target.value)}
              disabled={uploading}
            >
              {(pretrains.length ? pretrains : [{ id: "standard", name: "RVC 標準", sample_rates: [sampleRate] }]).map(
                (item) => (
                  <option key={item.id} value={item.id} disabled={!item.sample_rates.includes(sampleRate)}>
                    {item.name}
                    {item.sample_rates.includes(sampleRate) ? "" : `（${sampleRate / 1000}kHz は未対応）`}
                  </option>
                ),
              )}
            </select>
          </Field>

          <details className="rounded-xl border border-zinc-200 px-4 py-3 dark:border-zinc-800">
            <summary className="cursor-pointer text-sm font-medium select-none">
              詳しい設定（ふつうは変えなくて大丈夫です）
            </summary>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <Field label="サンプルレート" htmlFor={ids.rate} hint="40kHz がおすすめです">
                <select
                  id={ids.rate}
                  className={inputClass}
                  value={sampleRate}
                  onChange={(event) => setSampleRate(Number(event.target.value))}
                  disabled={uploading}
                >
                  <option value={40000}>40kHz（おすすめ）</option>
                  <option value={48000}>48kHz（高音質・少し重い）</option>
                  <option value={32000}>32kHz（軽い）</option>
                </select>
              </Field>
              <Field
                label="バッチサイズ"
                htmlFor={ids.batch}
                hint={`この PC のおすすめは ${system?.recommended_batch_size ?? 4} です。メモリ不足で止まるときは小さくしてください`}
              >
                <input
                  id={ids.batch}
                  type="number"
                  min={1}
                  max={32}
                  className={inputClass}
                  value={batchSize}
                  onChange={(event) => setChosenBatch(Number(event.target.value) || 1)}
                  disabled={uploading}
                />
              </Field>
              <Field
                label="保存の間隔（エポック）"
                htmlFor={ids.save}
                hint="この間隔で途中のモデルを保存します。止まっても続きから再開できます"
              >
                <input
                  id={ids.save}
                  type="number"
                  min={1}
                  max={100}
                  className={inputClass}
                  value={saveEvery}
                  onChange={(event) => setSaveEvery(Number(event.target.value) || 1)}
                  disabled={uploading}
                />
              </Field>
              <Field
                label="声の特徴を取り出すモデル"
                htmlFor={ids.embedder}
                hint="ふつうは contentvec のままにしてください。事前学習モデルは contentvec で作られているものがほとんどです"
              >
                <select
                  id={ids.embedder}
                  className={inputClass}
                  value={embedder}
                  onChange={(event) => setEmbedder(event.target.value as Embedder)}
                  disabled={uploading}
                >
                  {(system?.embedders ?? (Object.keys(EMBEDDER_LABELS) as Embedder[])).map((item) => (
                    <option key={item} value={item}>
                      {EMBEDDER_LABELS[item] ?? item}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
          </details>
        </div>

        {error && (
          <Alert tone="danger" title="学習を始められませんでした">
            {error}
          </Alert>
        )}
        {startedName && !error && (
          <Alert tone="success" title={`「${startedName}」の学習を始めました`}>
            進み具合はこのページの上に表示されます。ほかのページに移動しても学習は続きます。
            動画の音声のダウンロードや、BGM・伴奏の除去がある場合は、それが終わってから学習が始まります。
            学習の途中で保存されたモデルは、モデル一覧で試し聴きできます。
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

        <Button type="submit" disabled={!canSubmit}>
          {uploading ? "送っています…" : "学習を始める"}
        </Button>
      </form>
    </Card>
  );
}
