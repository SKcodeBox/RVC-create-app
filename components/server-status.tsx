"use client";

import { createContext, use, type ReactNode } from "react";
import { api, type SystemInfo } from "@/lib/api";
import { usePolling } from "@/lib/hooks";
import { Alert, Code } from "./ui";

type State = "checking" | "offline" | "setup" | "ready";

interface ServerStatus {
  state: State;
  system: SystemInfo | null;
  refresh: () => Promise<unknown>;
}

const ServerStatusContext = createContext<ServerStatus>({
  state: "checking",
  system: null,
  refresh: async () => undefined,
});

/** RVC サーバーの状態を数秒おきに確認し、画面全体で共有する。 */
export function ServerStatusProvider({ children }: { children: ReactNode }) {
  const { data, error, refresh } = usePolling(api.system, 4000);
  const state: State = error
    ? "offline"
    : data
      ? data.ready
        ? "ready"
        : "setup"
      : "checking";
  return (
    <ServerStatusContext value={{ state, system: error ? null : (data ?? null), refresh }}>
      {children}
    </ServerStatusContext>
  );
}

export function useServerStatus() {
  return use(ServerStatusContext);
}

const DEVICE_LABELS: Record<string, string> = {
  mps: "Mac の GPU",
  cuda: "NVIDIA GPU",
  cpu: "CPU",
};

export function ServerStatusPill() {
  const { state, system } = useServerStatus();
  const styles: Record<State, { dot: string; text: string }> = {
    checking: { dot: "bg-zinc-400", text: "確認中…" },
    offline: { dot: "bg-red-500", text: "サーバー停止中" },
    setup: { dot: "bg-amber-500", text: "準備が必要" },
    ready: {
      dot: "bg-emerald-500",
      text: `準備OK${
        system && DEVICE_LABELS[system.device]
          ? `（${system.device === "cuda" && system.device_name ? system.device_name : DEVICE_LABELS[system.device]}）`
          : ""
      }`,
    },
  };
  const { dot, text } = styles[state];
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-zinc-200 px-3 py-1 text-xs text-zinc-600 dark:border-zinc-800 dark:text-zinc-300"
      title="RVC サーバーの状態"
    >
      <span className={`size-2 rounded-full ${dot}`} aria-hidden="true" />
      {text}
    </span>
  );
}

/** サーバーが止まっている・準備が終わっていないときの案内。問題がなければ何も出さない。 */
export function ServerNotice() {
  const { state, system } = useServerStatus();
  if (state === "offline") {
    return (
      <div className="mb-6">
        <Alert tone="danger" title="RVC サーバーに接続できません">
          <p>
            学習や変換は、この PC で動く RVC サーバーが行います。ターミナル（Windows は PowerShell
            またはコマンドプロンプト）でプロジェクトのフォルダを開き、次のどちらかを実行してください。
          </p>
          <ul className="list-disc space-y-1 pl-5">
            <li>
              画面とサーバーをまとめて起動: <Code>npm run dev:all</Code>
            </li>
            <li>
              サーバーだけ起動: <Code>npm run rvc:server</Code>
            </li>
          </ul>
          <p>初めて使うときは、先に <Code>npm run rvc:setup</Code> で準備が必要です。</p>
        </Alert>
      </div>
    );
  }
  if (state === "setup" && system) {
    return (
      <div className="mb-6">
        <Alert tone="warning" title="セットアップが終わっていません">
          <p>不足しているもの: {system.missing.join("、")}</p>
          <p>
            ターミナルで <Code>npm run rvc:setup</Code> を実行してから、サーバーを起動し直してください。
          </p>
        </Alert>
      </div>
    );
  }
  return null;
}
