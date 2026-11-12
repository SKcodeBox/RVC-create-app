"use client";

import { useState, type ReactNode } from "react";
import { Button } from "./ui";

/** 押すと確認が出て、もう一度押すと実行するボタン（削除や中止の押し間違いを防ぐ）。 */
export function ConfirmButton({
  children,
  question,
  confirmLabel,
  onConfirm,
  disabled = false,
}: {
  children: ReactNode;
  question: string;
  confirmLabel: string;
  onConfirm: () => Promise<unknown> | void;
  disabled?: boolean;
}) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);

  if (!asking) {
    return (
      <Button variant="secondary" size="sm" disabled={disabled} onClick={() => setAsking(true)}>
        {children}
      </Button>
    );
  }

  return (
    <span className="inline-flex flex-wrap items-center gap-2" role="group">
      <span className="text-sm text-zinc-700 dark:text-zinc-300">{question}</span>
      <Button
        variant="danger"
        size="sm"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          try {
            await onConfirm();
          } finally {
            setBusy(false);
            setAsking(false);
          }
        }}
      >
        {confirmLabel}
      </Button>
      <Button variant="ghost" size="sm" disabled={busy} onClick={() => setAsking(false)}>
        やめる
      </Button>
    </span>
  );
}
