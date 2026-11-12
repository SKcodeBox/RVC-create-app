// スクリプト共通の環境変数。
// 使っているライブラリ・ツールが開発元へ利用状況を送る機能（テレメトリ）は、アプリの動きに関係ない。
// 送信の機能だけを持つライブラリ（onnxruntime など）は入れず、必要なライブラリの中にあるものはここで止める。
export const NO_TELEMETRY_ENV = {
  NEXT_TELEMETRY_DISABLED: "1", // Next.js → Vercel
  HF_HUB_DISABLE_TELEMETRY: "1", // Hugging Face のライブラリ（Applio が声の特徴を取り出すのに使う）
  DO_NOT_TRACK: "1", // この取り決めに従うツール全般
};

/** 子プロセスに渡す環境変数（テレメトリを止め、Python が日本語を正しく扱えるようにする）。 */
export function childEnv(extra = {}) {
  return { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8", ...NO_TELEMETRY_ENV, ...extra };
}
