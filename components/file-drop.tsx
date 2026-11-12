"use client";

import { useId, useState, type DragEvent } from "react";
import { UploadIcon } from "./icons";

/** クリックで選ぶか、ドラッグ＆ドロップでファイルを受け取る場所。 */
export function FileDrop({
  accept,
  multiple = false,
  disabled = false,
  title,
  hint,
  onFiles,
}: {
  accept: string;
  multiple?: boolean;
  disabled?: boolean;
  title: string;
  hint?: string;
  onFiles: (files: File[]) => void;
}) {
  const id = useId();
  const [dragging, setDragging] = useState(false);

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    if (disabled) return;
    const files = Array.from(event.dataTransfer.files);
    if (files.length > 0) onFiles(multiple ? files : files.slice(0, 1));
  }

  return (
    <label
      htmlFor={id}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors ${
        dragging
          ? "border-violet-500 bg-violet-50 dark:bg-violet-500/10"
          : "border-zinc-300 hover:border-violet-400 hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800/50"
      } ${disabled ? "pointer-events-none opacity-60" : ""}`}
    >
      <UploadIcon width={28} height={28} className="text-violet-600 dark:text-violet-400" />
      <span className="font-medium">{title}</span>
      {hint && <span className="text-xs text-zinc-500 dark:text-zinc-400">{hint}</span>}
      <input
        id={id}
        type="file"
        accept={accept}
        multiple={multiple}
        disabled={disabled}
        className="sr-only"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          if (files.length > 0) onFiles(files);
          // 同じファイルをもう一度選べるようにする
          event.target.value = "";
        }}
      />
    </label>
  );
}
