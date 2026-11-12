// 動作の確認。Windows・macOS・Linux 共通。
//   npm test          … サーバーの自動テスト（pytest）・静的チェック（pyflakes）・画面の型チェックと lint
//   npm run test:e2e  … 上に加えて、本物の AI で学習・変換を通しで試す（数分〜十数分）
import { spawnSync } from "node:child_process";
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
const e2e = process.argv.includes("--e2e");

if (!existsSync(python)) {
  console.error("RVC サーバーの準備がまだです。先に npm run rvc:setup を実行してください。");
  process.exit(1);
}

const results = [];
function run(label, command, args, cwd = root) {
  console.log(`\n\x1b[1;36m==> ${label}\x1b[0m`);
  // next dev / next build と同じく、AI エージェントの検出で余計なファイルが書かれないようにする
  const env = childEnv();
  for (const key of ["AI_AGENT", "CLAUDECODE", "CLAUDE_CODE"]) delete env[key];
  const result = spawnSync(command, args, { cwd, stdio: "inherit", env, shell: isWindows && command === "npx" });
  results.push([label, result.status === 0]);
}

run("サーバーの静的チェック（pyflakes）", python, ["-m", "pyflakes", "rvc_api", "engine", "separation/separate.py", "patches", "tests"], serverDir);
run("サーバーの自動テスト（pytest）", python, ["-m", "pytest", "tests", "-q"], serverDir);
run("画面の型チェック（tsc）", "npx", ["tsc", "--noEmit"]);
run("画面の lint（eslint）", "npx", ["eslint"]);
if (e2e) run("通しの動作確認（学習・変換）", python, [path.join("tests", "e2e_smoke.py")], serverDir);

console.log("\n結果:");
for (const [label, ok] of results) console.log(`  ${ok ? "\x1b[32mOK\x1b[0m  " : "\x1b[31mNG\x1b[0m  "}${label}`);
process.exit(results.every(([, ok]) => ok) ? 0 : 1);
