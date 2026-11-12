// RVC サーバーの準備（最初に1回だけ実行する。やり直しても問題ない）。Windows・macOS・Linux 共通。
//   使い方: npm run rvc:setup
//   1. uv（Python の管理ツール）を用意する
//   2. Applio（RVC の本体）をダウンロードして、このアプリ用の修正を当てる
//   3. Python 3.12 の仮想環境を作り、必要なライブラリを入れる（NVIDIA の GPU があれば GPU 版の PyTorch）
//   4. BGM・伴奏を取り除く機能（ボーカル分離）用の仮想環境を作る
//   5. 学習・変換に使う AI モデルをダウンロードする（約1GB）
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, renameSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { childEnv } from "./env.mjs";

const APPLIO_VERSION = "3.7.1";
const APPLIO_TARBALL = `https://github.com/IAHispano/Applio/archive/refs/tags/${APPLIO_VERSION}.tar.gz`;
// NVIDIA の GPU 用の PyTorch の配布場所。Applio 本体は Applio と同じ CUDA 12.8 版、
// ボーカル分離は、使う PyTorch 2.14.1 に CUDA 12.8 版がないため CUDA 12.6 版にする
const TORCH_INDEX_APPLIO = "https://download.pytorch.org/whl/cu128";
const TORCH_INDEX_SEPARATION = "https://download.pytorch.org/whl/cu126";

const isWindows = process.platform === "win32";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const serverDir = path.join(root, "rvc-server");
const applioDir = path.join(serverDir, "applio");
const dataDir = path.join(serverDir, "data");

const venvPython = (venv) =>
  isWindows ? path.join(venv, "Scripts", "python.exe") : path.join(venv, "bin", "python");
const mainPython = venvPython(path.join(serverDir, ".venv"));
const separationPython = venvPython(path.join(serverDir, ".venv-separation"));

function step(message) {
  console.log(`\n\x1b[1;36m==> ${message}\x1b[0m`);
}

function fail(message) {
  console.error(`\n\x1b[1;31mセットアップを続けられません:\x1b[0m ${message}`);
  process.exit(1);
}

/** コマンドを実行する（出力はそのまま表示）。失敗したら止める。 */
function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: "inherit",
    cwd: options.cwd ?? serverDir,
    env: childEnv({ UV_HTTP_TIMEOUT: "300", ...options.env }),
    shell: false,
  });
  if (result.error) fail(`${command} を実行できませんでした（${result.error.message}）`);
  if (result.status !== 0 && !options.allowFailure) {
    fail(options.failMessage ?? `${path.basename(command)} ${args[0] ?? ""} が失敗しました（終了コード ${result.status}）`);
  }
  return result.status === 0;
}

/** コマンドを実行して、出力を文字列で受け取る（なければ null）。 */
function capture(command, args) {
  const result = spawnSync(command, args, { encoding: "utf8", shell: false });
  if (result.error || result.status !== 0) return null;
  return result.stdout.trim();
}

function findUv() {
  const candidates = [
    process.env.RVC_UV,
    path.join(os.homedir(), ".local", "bin", isWindows ? "uv.exe" : "uv"),
  ].filter(Boolean);
  if (capture("uv", ["--version"]) !== null) return "uv";
  return candidates.find((candidate) => existsSync(candidate)) ?? null;
}

function installUv() {
  step("uv（Python の管理ツール）をインストールします");
  // シェルの設定ファイル（PATH）は書き換えない
  if (isWindows) {
    run("powershell", [
      "-NoProfile",
      "-ExecutionPolicy",
      "ByPass",
      "-Command",
      "$env:UV_NO_MODIFY_PATH=1; irm https://astral.sh/uv/install.ps1 | iex",
    ]);
  } else {
    run("sh", ["-c", "curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh"]);
  }
  return findUv() ?? fail("uv のインストールに失敗しました。https://docs.astral.sh/uv/ の手順で入れてください。");
}

async function downloadApplio() {
  step(`Applio ${APPLIO_VERSION} をダウンロードします`);
  const response = await fetch(APPLIO_TARBALL);
  if (!response.ok) fail(`Applio をダウンロードできませんでした（${response.status}）。インターネット接続を確認してください。`);
  const temp = mkdtempSync(path.join(os.tmpdir(), "rvc-studio-"));
  const archive = path.join(temp, "applio.tar.gz");
  writeFileSync(archive, Buffer.from(await response.arrayBuffer()));
  // tar は macOS・Linux・Windows 10 以降に標準で入っている
  run("tar", ["-xzf", archive, "-C", temp], { failMessage: "Applio の展開に失敗しました。" });
  const extracted = path.join(temp, `Applio-${APPLIO_VERSION}`);
  if (!existsSync(path.join(extracted, "core.py"))) fail("ダウンロードした Applio の中身が想定と違います。");
  rmSync(applioDir, { recursive: true, force: true });
  renameSync(extracted, applioDir);
  rmSync(temp, { recursive: true, force: true });
}

/** NVIDIA の GPU が使えるか（ドライバーの nvidia-smi で調べる）。macOS では使わない。 */
function nvidiaGpu() {
  if (process.platform === "darwin") return null;
  const output = capture("nvidia-smi", ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]);
  if (!output) return null;
  const [name, driver, memory] = output.split("\n")[0].split(",").map((part) => part.trim());
  return { name, driver, memory };
}

function cudaArgs(index) {
  return ["--extra-index-url", index, "--index-strategy", "unsafe-best-match"];
}

// 前のバージョンのセットアップで入れていた、もう使わないライブラリ。
// onnxruntime は使わない機能専用で、Microsoft へ利用状況を送る処理を含む。
// Windows の NVIDIA の部品は onnxruntime-gpu だけが使っていた（PyTorch は自分の分を持っている。
// Linux の PyTorch は同じ名前の部品を使うので、Windows 以外では消さない）
const UNUSED_MAIN = [
  "onnxruntime",
  "onnxruntime-gpu",
  ...(isWindows ? ["nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12", "nvidia-cufft-cu12"] : []),
];
const UNUSED_SEPARATION = ["onnxruntime", "onnxruntime-gpu"];

/** 入っていれば削除する。 */
function removeUnused(uv, python, packages) {
  const listed = capture(uv, ["pip", "list", "--python", python, "--format", "json"]);
  if (listed === null) return;
  const installed = new Set(JSON.parse(listed).map((item) => item.name.toLowerCase()));
  const targets = packages.filter((name) => installed.has(name));
  if (targets.length === 0) return;
  console.log(`使わないライブラリを削除します: ${targets.join(", ")}`);
  run(uv, ["pip", "uninstall", "--python", python, ...targets]);
}

async function main() {
  console.log(`RVC Studio のセットアップを始めます（${process.platform} / ${process.arch}）`);
  if (process.platform === "darwin" && process.arch !== "arm64") {
    console.warn("注意: Intel の Mac では GPU を使えないため、学習にとても時間がかかります。");
  }

  const uv = findUv() ?? installUv();
  console.log(`uv: ${capture(uv, ["--version"]) ?? uv}`);

  if (!existsSync(path.join(applioDir, "core.py"))) await downloadApplio();

  if (!existsSync(mainPython)) {
    step("Python 3.12 の仮想環境を作ります");
    run(uv, ["venv", ".venv", "--python", "3.12"]);
  }

  step("Applio に、このアプリ用の修正を当てます");
  run(mainPython, [path.join(serverDir, "patches", "apply_patches.py"), applioDir], {
    failMessage: "Applio に修正を当てられませんでした。rvc-server/applio を削除して、もう一度実行してください。",
  });

  const gpu = nvidiaGpu();
  if (gpu) {
    console.log(`NVIDIA の GPU が見つかりました: ${gpu.name}（ドライバー ${gpu.driver}・${gpu.memory}）。GPU 版の PyTorch を入れます。`);
    const major = Number.parseInt(gpu.driver, 10);
    if (Number.isFinite(major) && major < 570) {
      console.warn("注意: GPU のドライバーが古いと GPU を使えないことがあります。NVIDIA のサイトから最新版に更新してください。");
    }
  } else if (process.platform !== "darwin") {
    console.warn("注意: NVIDIA の GPU が見つかりませんでした。CPU で動くため、学習にとても時間がかかります。");
  }

  step("必要なライブラリをインストールします（数分かかります）");
  // yt-dlp は YouTube の仕様変更に合わせて頻繁に更新されるため、毎回最新版にする
  run(uv, [
    "pip", "install", "--python", mainPython,
    "-r", "requirements.txt",
    "--upgrade-package", "yt-dlp", "--upgrade-package", "yt-dlp-ejs",
    ...(gpu ? cudaArgs(TORCH_INDEX_APPLIO) : []),
  ]);
  removeUnused(uv, mainPython, UNUSED_MAIN);

  step("BGM・伴奏を取り除く機能（ボーカル分離）のライブラリをインストールします（約1〜3GB）");
  // audio-separator は Applio より新しい PyTorch が必要なため、別の仮想環境に入れる。
  // 分離に使う AI モデル（約1.5GB〜）は、初めて使うときに自動でダウンロードする
  if (!existsSync(separationPython)) run(uv, ["venv", ".venv-separation", "--python", "3.12"]);
  run(uv, [
    "pip", "install", "--python", separationPython,
    "-r", "requirements-separation.txt", "--override", "separation-overrides.txt",
    ...(gpu ? cudaArgs(TORCH_INDEX_SEPARATION) : []),
  ]);
  removeUnused(uv, separationPython, UNUSED_SEPARATION);

  step("AI モデルをダウンロードします（約1GB）");
  mkdirSync(path.join(dataDir, "models"), { recursive: true });
  mkdirSync(path.join(dataDir, "applio-config"), { recursive: true });
  const applioEnv = {
    APPLIO_LOGS_DIR: path.join(dataDir, "models"),
    APPLIO_CONFIG_DIR: path.join(dataDir, "applio-config"),
    PYTHONPATH: applioDir,
  };
  // ピッチ検出（rmvpe など）と、声の特徴を取り出すモデル（contentvec）
  run(mainPython, ["core.py", "prerequisites", "--no-pretraineds-hifigan", "--models", "--no-exe"], {
    cwd: applioDir,
    env: applioEnv,
  });
  // 学習の出発点になる事前学習モデル（40kHz 用）。他のサンプルレートは学習時に自動で入る
  run(mainPython, [path.join("rvc", "lib", "tools", "prerequisites_download.py"), "--vocoder", "HiFi-GAN", "--sample-rate", "40000"], {
    cwd: applioDir,
    env: applioEnv,
  });

  step("計算に使う装置を確かめます");
  run(mainPython, [
    "-c",
    "import torch; print('NVIDIA の GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Mac の GPU（MPS）' if torch.backends.mps.is_available() else 'CPU（GPU は使えません）')",
  ], { allowFailure: true });

  step("準備ができました");
  console.log("  起動するには: npm run app （画面と RVC サーバーをまとめて起動。http://127.0.0.1:3000 を開く）");
  console.log("  動作の確認: npm test");
}

await main();
