// 画面（Next.js）と RVC サーバー（Python）をまとめて起動する。Ctrl+C で両方止まる。Windows・macOS・Linux 共通。
//   使い方: npm run app      … 普段使い（画面を本番用にビルドして起動。軽くて速い）
//           npm run dev:all  … 開発用（コードを書き換えるとすぐ画面に反映される。メモリを多く使う）
import { spawn, spawnSync } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { childEnv } from "./env.mjs";
import { nextArgs } from "./next.mjs";

const isWindows = process.platform === "win32";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RESET = "\x1b[0m";
const production = process.argv.includes("--prod");

/** フォルダの中で一番新しいファイルの更新時刻。 */
function newestMtime(target) {
  if (!existsSync(target)) return 0;
  const stat = statSync(target);
  if (!stat.isDirectory()) return stat.mtimeMs;
  let newest = 0;
  for (const entry of readdirSync(target)) newest = Math.max(newest, newestMtime(path.join(target, entry)));
  return newest;
}

/** 本番用のビルドが、画面のコードより古ければ（またはなければ）ビルドし直す。 */
function buildIfNeeded() {
  const buildId = path.join(root, ".next", "BUILD_ID");
  const sources = ["app", "components", "lib", "public", "next.config.ts", "package.json", "package-lock.json", "tsconfig.json", "postcss.config.mjs"];
  const newest = Math.max(...sources.map((item) => newestMtime(path.join(root, item))));
  if (existsSync(buildId) && statSync(buildId).mtimeMs > newest) return;
  console.log("画面を本番用にビルドしています（画面のコードを変えたときだけ。1分ほどかかります）…");
  const result = spawnSync(process.execPath, nextArgs("build"), { cwd: root, stdio: "inherit", env: childEnv() });
  if (result.status !== 0) {
    console.error("画面のビルドに失敗しました。上のエラーを確認してください（開発用の npm run dev:all なら起動できる場合があります）。");
    process.exit(result.status ?? 1);
  }
}

const services = [
  {
    name: "rvc",
    color: "\x1b[35m",
    command: process.execPath,
    args: [path.join(root, "scripts", "rvc-server.mjs")],
  },
  {
    name: "web",
    color: "\x1b[36m",
    command: process.execPath,
    args: nextArgs(production ? "start" : "dev"),
  },
];

const python = isWindows
  ? path.join(root, "rvc-server", ".venv", "Scripts", "python.exe")
  : path.join(root, "rvc-server", ".venv", "bin", "python");
if (!existsSync(python)) {
  console.warn("RVC サーバーの準備がまだです。先に npm run rvc:setup を実行してください。");
}
if (production) buildIfNeeded();

function pipeWithPrefix(stream, { name, color }, target) {
  let buffer = "";
  stream.setEncoding("utf8");
  stream.on("data", (chunk) => {
    buffer += chunk;
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? "";
    for (const line of lines) target.write(`${color}[${name}]${RESET} ${line}\n`);
  });
  stream.on("end", () => {
    if (buffer) target.write(`${color}[${name}]${RESET} ${buffer}\n`);
  });
}

const running = new Map();
let stopping = false;

/** Windows で、プロセスとその子プロセスをまとめて強制終了する。 */
function killTree(child) {
  spawnSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], { stdio: "ignore" });
}

function stopAll() {
  if (stopping) return;
  stopping = true;
  for (const child of running.values()) {
    if (isWindows) {
      // Windows ではコンソールの Ctrl+C が子プロセスにも直接届く。止まらないものだけ後で強制終了する
      continue;
    }
    // 子プロセスは別のプロセスグループで動かしているので、合図は1回だけ届く
    // （RVC サーバーは2回目の Ctrl+C で後片付けを省いて終了してしまうため）
    child.kill("SIGINT");
  }
  // 30秒たっても止まらないものは強制終了する
  setTimeout(() => {
    for (const child of running.values()) {
      if (isWindows) killTree(child);
      else child.kill("SIGKILL");
    }
    process.exit(1);
  }, 30_000).unref();
}

for (const service of services) {
  const child = spawn(service.command, service.args, {
    cwd: root,
    env: childEnv({ FORCE_COLOR: "1" }),
    stdio: ["ignore", "pipe", "pipe"],
    // Windows で detached にすると新しいコンソール窓が開いてしまうため、macOS・Linux だけにする
    detached: !isWindows,
    windowsHide: true,
  });
  running.set(service.name, child);
  pipeWithPrefix(child.stdout, service, process.stdout);
  pipeWithPrefix(child.stderr, service, process.stderr);
  child.on("exit", (code, signal) => {
    running.delete(service.name);
    if (stopping) {
      if (running.size === 0) process.exit(0);
      return;
    }
    console.log(`${service.color}[${service.name}]${RESET} 停止しました（${signal ?? code}）`);
    if (service.name === "web") {
      stopAll();
    } else {
      console.log(
        `${service.color}[${service.name}]${RESET} 画面はそのまま動いています。RVC サーバーだけ起動し直すには npm run rvc:server`,
      );
    }
  });
}

process.on("SIGINT", stopAll);
process.on("SIGTERM", stopAll);
