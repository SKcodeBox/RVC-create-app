"use client";

import { useId, useState, type FormEvent } from "react";
import { modelNameError, upload, type VoiceType } from "@/lib/api";
import { useServerStatus } from "@/components/server-status";
import { Alert, Button, buttonClass, Field, inputClass, ProgressBar } from "@/components/ui";

/** ファイルを1つ選ぶ欄（ブラウザ標準の欄は英語で表示されることがあるため、自前で表示する）。 */
function FilePicker({
  id,
  accept,
  file,
  onChange,
}: {
  id: string;
  accept: string;
  file: File | null;
  onChange: (file: File | null) => void;
}) {
  return (
    <div className="flex min-w-0 items-center gap-3">
      <input
        id={id}
        type="file"
        accept={accept}
        className="peer sr-only"
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
      />
      <label
        htmlFor={id}
        className={`${buttonClass("secondary", "sm")} shrink-0 cursor-pointer peer-focus-visible:ring-2 peer-focus-visible:ring-violet-500 peer-focus-visible:ring-offset-2 dark:peer-focus-visible:ring-offset-zinc-900`}
      >
        ファイルを選ぶ
      </label>
      <span className="min-w-0 truncate text-sm text-zinc-600 dark:text-zinc-400" title={file?.name}>
        {file ? file.name : "選ばれていません"}
      </span>
    </div>
  );
}

export function ImportForm({ onImported }: { onImported: () => void }) {
  const ids = { name: useId(), pth: useId(), index: useId(), voice: useId() };
  const { state } = useServerStatus();
  const [name, setName] = useState("");
  const [pth, setPth] = useState<File | null>(null);
  const [index, setIndex] = useState<File | null>(null);
  const [voiceType, setVoiceType] = useState<VoiceType | "">("");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const [formKey, setFormKey] = useState(0);

  const nameError = modelNameError(name);
  const canSubmit = state === "ready" && name && !nameError && pth && progress === null;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit || !pth) return;
    setError(null);
    setDone(null);
    const form = new FormData();
    form.set("name", name);
    form.set("model_file", pth, pth.name);
    if (index) form.set("index_file", index, index.name);
    if (voiceType) form.set("voice_type", voiceType);
    setProgress(0);
    try {
      await upload("/api/models/import", form, setProgress);
      setDone(name);
      setName("");
      setPth(null);
      setIndex(null);
      setVoiceType("");
      // ファイル選択欄を空に戻す
      setFormKey((key) => key + 1);
      onImported();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setProgress(null);
    }
  }

  return (
    <form key={formKey} onSubmit={handleSubmit} className="space-y-4">
      <p className="text-sm text-zinc-600 dark:text-zinc-400">
        ほかの RVC アプリで作ったモデル（.pth）を、このアプリで使えるようにします。
        インデックス（.index）もあれば一緒に選ぶと、より本人らしい声になります。
      </p>
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="モデル名" htmlFor={ids.name} hint="例: friend-voice" error={nameError}>
          <input
            id={ids.name}
            className={inputClass}
            value={name}
            onChange={(event) => setName(event.target.value.trim())}
            placeholder="friend-voice"
            maxLength={40}
            autoComplete="off"
            aria-invalid={nameError ? true : undefined}
          />
        </Field>
        <Field label="モデルファイル（.pth）" htmlFor={ids.pth}>
          <FilePicker id={ids.pth} accept=".pth" file={pth} onChange={setPth} />
        </Field>
        <Field label="インデックス（.index・なくても可）" htmlFor={ids.index}>
          <FilePicker id={ids.index} accept=".index" file={index} onChange={setIndex} />
        </Field>
        <Field
          label="このモデルの声の高さ"
          htmlFor={ids.voice}
          hint="選ぶと、変換のときに声の高さを自動で合わせられます（あとからモデル一覧でも変えられます）"
        >
          <select
            id={ids.voice}
            className={inputClass}
            value={voiceType}
            onChange={(event) => setVoiceType(event.target.value as VoiceType | "")}
          >
            <option value="">わからない</option>
            <option value="low">低め（男性の声など）</option>
            <option value="high">高め（女性の声など）</option>
          </select>
        </Field>
      </div>
      {error && (
        <Alert tone="danger" title="取り込めませんでした">
          {error}
        </Alert>
      )}
      {done && <Alert tone="success">「{done}」を取り込みました。</Alert>}
      {progress !== null && <ProgressBar value={progress} label="取り込みの進み具合" />}
      <Button type="submit" disabled={!canSubmit}>
        取り込む
      </Button>
    </form>
  );
}
