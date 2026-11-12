// Next.js（画面）を起動する。Windows・macOS・Linux 共通。
//   使い方: node scripts/next.mjs <dev|build|start> [Next.js のオプション]
// - 開発元（Vercel）への利用状況の送信を止める
// - dev / start は、この PC からだけ開けるようにする（同じ Wi-Fi の他の機器からは開けない）
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { childEnv } from "./env.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const nextBin = path.join(root, "node_modules", "next", "dist", "bin", "next");

/** next の引数（dev / start なら、指定がなければ 127.0.0.1 だけで待ち受ける）。 */
export function nextArgs(command, extra = []) {
  const args = [nextBin, command, ...extra];
  const hasHost = extra.some((arg) => arg === "-H" || arg === "--hostname" || arg.startsWith("--hostname="));
  if ((command === "dev" || command === "start") && !hasHost) args.push("-H", "127.0.0.1");
  return args;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [command = "dev", ...extra] = process.argv.slice(2);
  const child = spawn(process.execPath, nextArgs(command, extra), { cwd: root, stdio: "inherit", env: childEnv() });
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.on(signal, () => {
      // Windows ではコンソールの Ctrl+C が子プロセスにも直接届く
      if (process.platform !== "win32") child.kill(signal);
    });
  }
  child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
}
