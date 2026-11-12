// RVC サーバー（API）を起動する。Windows・macOS・Linux 共通。終了は Ctrl+C。
//   使い方: npm run rvc:server
//   環境変数: RVC_HOST（初期値 127.0.0.1）・RVC_PORT（初期値 8765）・RVC_ACCESS_LOG=1 でアクセスログを出す
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { childEnv } from "./env.mjs";

const isWindows = process.platform === "win32";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const serverDir = path.join(root, "rvc-server");
const python = isWindows
  ? path.join(serverDir, ".venv", "Scripts", "python.exe")
  : path.join(serverDir, ".venv", "bin", "python");

if (!existsSync(python) || !existsSync(path.join(serverDir, "applio", "core.py"))) {
  console.error("RVC サーバーの準備がまだです。先に npm run rvc:setup を実行してください。");
  process.exit(1);
}

const host = process.env.RVC_HOST ?? "127.0.0.1";
const port = process.env.RVC_PORT ?? "8765";
const args = ["-m", "uvicorn", "rvc_api.main:app", "--host", host, "--port", port];
// 画面が数秒おきに状態を問い合わせるため、アクセスログは普段は出さない
if (process.env.RVC_ACCESS_LOG !== "1") args.push("--no-access-log");

console.log(`RVC サーバーを起動します: http://${host}:${port} （終了は Ctrl+C）`);
const child = spawn(python, args, {
  cwd: serverDir,
  stdio: "inherit",
  env: childEnv(),
});

let stopping = false;
function stop(signal) {
  if (stopping) return;
  stopping = true;
  // Windows ではコンソールの Ctrl+C がサーバーにも直接届くので、ここでは送らない
  if (!isWindows) child.kill(signal);
  // 学習中のプロセスの後片付けを待つ。30秒たっても止まらなければ強制終了する
  setTimeout(() => {
    child.kill("SIGKILL");
    process.exit(1);
  }, 30_000).unref();
}

process.on("SIGINT", () => stop("SIGINT"));
process.on("SIGTERM", () => stop("SIGTERM"));
child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
