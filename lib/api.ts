// RVC サーバー（rvc-server/、FastAPI）とやり取りするための型と関数。
// ブラウザから直接サーバーにアクセスする。サーバーの場所は NEXT_PUBLIC_RVC_API_URL で変えられる。

export const API_URL = (
  process.env.NEXT_PUBLIC_RVC_API_URL ?? "http://127.0.0.1:8765"
).replace(/\/$/, "");

export type JobStatus =
  | "queued"
  | "running"
  | "done"
  | "error"
  | "cancelled"
  | "interrupted";

export const ACTIVE_STATUSES: readonly JobStatus[] = ["queued", "running"];

export function isActive(status: JobStatus): boolean {
  return ACTIVE_STATUSES.includes(status);
}

/** 学習に使う動画（YouTube など）。start / end は秒（null なら最初から・最後まで） */
export interface VideoSource {
  url: string;
  start: number | null;
  end: number | null;
}

/**
 * 学習の前に行う音声の下ごしらえ（サーバーの rvc_api/separation.py と合わせる）。
 * vocals: BGM・伴奏の除去 / dereverb: 残響・エコーの除去 / karaoke: ハモリの除去 / denoise: 雑音の除去
 */
export type PrepStep = "vocals" | "dereverb" | "karaoke" | "denoise";

/** AI モデルを使う下ごしらえ（初回にモデルのダウンロードが必要なもの） */
export type SeparationStep = Exclude<PrepStep, "denoise">;

/** BGM・伴奏の除去の精度。high: 2つの AI の組み合わせ（高精度）/ fast: 1つの AI（速さ優先） */
export type SeparationQuality = "high" | "fast";

/** 声の特徴を取り出すモデル（サーバーの rvc_api/library.py と合わせる） */
export type Embedder =
  | "contentvec"
  | "spin"
  | "spin-v2"
  | "japanese-hubert-base"
  | "chinese-hubert-base"
  | "korean-hubert-base";

export const EMBEDDER_LABELS: Record<Embedder, string> = {
  contentvec: "contentvec（標準・おすすめ）",
  spin: "spin",
  "spin-v2": "spin v2",
  "japanese-hubert-base": "日本語 HuBERT",
  "chinese-hubert-base": "中国語 HuBERT",
  "korean-hubert-base": "韓国語 HuBERT",
};

export interface TrainParams {
  name: string;
  sample_rate: number;
  total_epochs: number;
  batch_size: number;
  save_every: number;
  resume: boolean;
  dataset_files: number;
  dataset_duration: number;
  sources?: VideoSource[];
  prep?: PrepStep[];
  /** BGM・伴奏を取り除いたあと、声が入っている部分の合計（秒） */
  voiced_duration?: number;
  /** 以前の版の記録: 動画の音声の雑音を減らしたかどうか */
  cleanup?: boolean;
  embedder?: Embedder;
  pretrained?: string;
  early_stop?: boolean;
  separation_quality?: SeparationQuality;
}

/** 途中で保存したモデルの評価（検証用の音声をどれだけ再現できたか。小さいほど良い） */
export interface EpochEvaluation {
  epoch: number;
  val_loss: number | null;
  sample?: boolean;
}

export interface TrainDetail {
  start_epoch?: number;
  /** 終わったエポック数 */
  epoch?: number;
  /** いま学習しているエポック */
  current_epoch?: number;
  total_epochs?: number;
  step?: number;
  batch?: number | null;
  num_batches?: number | null;
  epoch_seconds?: number;
  eta_seconds?: number;
  /** 学習しすぎたら自動で止めるか（検証用の音声がないときは false） */
  early_stop?: boolean;
  validation_clips?: number;
  evaluations?: EpochEvaluation[];
  best_epoch?: number | null;
}

export type InputKind = "voice" | "song";
/** 曲のハモリの扱い。together: 分けずに一緒に変換 / convert: 分けてそれぞれ変換 / keep: 元の声のまま / remove: 消す */
export type HarmonyMode = "together" | "convert" | "keep" | "remove";
export type ExportFormat = "wav" | "flac" | "mp3";

export interface ConvertParams {
  model: string;
  weight: string;
  pitch: number;
  pitch_mode?: "manual" | "auto";
  f0_method: string;
  index_rate: number;
  protect: number;
  volume_envelope: number;
  autotune: boolean;
  autotune_strength?: number;
  formant?: boolean;
  formant_quefrency?: number;
  formant_timbre?: number;
  clean_input?: boolean;
  clean_output?: boolean;
  clean_strength?: number;
  export_format?: ExportFormat;
  input_kind?: InputKind;
  separation_quality?: SeparationQuality;
  dereverb?: boolean;
  harmony?: HarmonyMode;
  reverb?: number;
  vocal_gain_db?: number;
  instrumental_gain_db?: number;
  input_name: string;
  input_duration: number | null;
  upload_id: string;
}

export interface ConvertResult {
  output: string;
  duration: number | null;
  format?: ExportFormat;
  /** 曲の変換で作った「変換した声だけ」「伴奏だけ」の音声 */
  stems?: { vocals: string; instrumental: string } | null;
  pitch_used?: number;
  pitch_auto?: boolean;
  source_pitch?: number | null;
  target_pitch?: number | null;
}

interface JobBase {
  id: string;
  status: JobStatus;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  stage: string | null;
  stage_label: string | null;
  /** 全体の進み具合 0〜1（わからないときは null） */
  progress: number | null;
  message: string | null;
  logs: string[];
}

export interface TrainJob extends JobBase {
  kind: "train";
  params: TrainParams;
  detail: TrainDetail;
  result: {
    model: string;
    weight: string;
    index: string | null;
    recommended_epoch?: number | null;
    early_stopped_at?: number | null;
  } | null;
}

export interface ConvertJob extends JobBase {
  kind: "convert";
  params: ConvertParams;
  detail: { pitch_used?: number };
  result: ConvertResult | null;
}

export type Job = TrainJob | ConvertJob;

export interface ModelWeight {
  file: string;
  epoch: number | null;
  step: number | null;
  size: number;
  modified_at: number;
}

export interface VoiceModel {
  name: string;
  source: "trained" | "imported";
  created_at: number;
  weights: ModelWeight[];
  latest_weight: string | null;
  /** 変換に使うのにおすすめのモデルファイル（検証用の誤差がいちばん小さいもの） */
  recommended_weight?: string | null;
  recommended_epoch?: number | null;
  evaluated?: number;
  index_file: string | null;
  sample_rate: number | null;
  dataset_duration: number | null;
  trained_epochs: number;
  can_resume: boolean;
  /** 再開用のファイルが壊れている（保存の途中で止まったなど）。続きから学習はできない */
  checkpoint_broken?: boolean;
  /** 最後の学習が最後まで終わったか（キャンセル・中断・失敗なら false） */
  finished?: boolean;
  prep?: PrepStep[];
  embedder?: Embedder;
  pretrained?: string | null;
  early_stopped_at?: number | null;
  /** このモデルの声の高さの目安（Hz）。わからないときは null */
  target_pitch?: number | null;
  voice_type?: VoiceType | null;
  active_job_id: string | null;
  active_job_status: JobStatus | null;
  /** 学習用の音声（モデルを1つ取得したときだけ入る） */
  dataset?: DatasetFile[];
  /** 保存容量の内訳（モデルを1つ取得したときだけ入る） */
  storage?: ModelStorage;
}

/** 取り込んだモデルの声の高さ。low: 低め（男性の声など）/ high: 高め（女性の声など） */
export type VoiceType = "low" | "high";

export interface ModelStorage {
  weights: number;
  extra_weights: number;
  extra_weight_count: number;
  checkpoints: number;
  training_files: number;
  index: number;
  samples: number;
  dataset: number;
  stages: number;
}

/** 下ごしらえの途中経過。original: 下ごしらえの前 */
export type DatasetStage = "original" | "vocals" | "dereverb" | "karaoke";

export interface DatasetFile {
  file: string;
  size: number;
  duration: number | null;
  stages?: DatasetStage[];
}

export interface QualityInfo {
  /** Applio が学習中に記録した値（[エポック, 値] の並び） */
  curve: Partial<Record<"mel" | "generator" | "discriminator", [number, number][]>>;
  evaluations: {
    epoch: number;
    val_loss: number | null;
    has_sample: boolean;
    available: boolean;
    weight: string | null;
  }[];
  recommended_epoch: number | null;
  validation_clips: number | null;
  early_stopped_at: number | null;
}

export interface SeparationStepInfo {
  downloaded: boolean;
  download_mb: number;
  /** まだダウンロードしていない分（MB） */
  pending_mb: number;
  /** 処理にかかる時間の目安（音声の長さに対する倍率。Apple M5 で測った値） */
  speed: number;
  models: number;
}

export interface SeparationStatus {
  /** BGM・伴奏を取り除く機能の準備（npm run rvc:setup）ができているか */
  installed: boolean;
  steps: Record<SeparationStep, Record<SeparationQuality, SeparationStepInfo>>;
}

export interface PretrainedInfo {
  id: string;
  name: string;
  description: string;
  sample_rates: number[];
  downloaded: Record<string, boolean>;
}

export interface SystemInfo {
  ok: boolean;
  ready: boolean;
  missing: string[];
  applio_version: string | null;
  platform?: "mac" | "windows" | "linux";
  device: string;
  device_name?: string | null;
  device_memory_gb?: number | null;
  recommended_batch_size?: number;
  cpu_count: number;
  disk_free_bytes: number;
  ffmpeg?: boolean;
  accepted_extensions?: string[];
  pretrained: Record<string, boolean>;
  pretrains?: PretrainedInfo[];
  embedders?: Embedder[];
  /** 古いサーバーでは入っていない */
  separation?: SeparationStatus;
  active_jobs: Job[];
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const OFFLINE_MESSAGE =
  "RVC サーバーに接続できません。サーバーが起動しているか確認してください。";

function messageFromBody(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return "入力内容に誤りがあります。";
  }
  return `エラーが発生しました（${status}）。`;
}

/**
 * データを変える操作に付ける専用ヘッダー（サーバーの rvc_api/main.py と合わせる）。
 * ほかのサイトから勝手に操作されるのを防ぐためのもの。
 */
const CLIENT_HEADER = "X-RVC-Client";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  const headers = new Headers(init?.headers);
  if ((init?.method ?? "GET").toUpperCase() !== "GET") headers.set(CLIENT_HEADER, "1");
  try {
    response = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init, headers });
  } catch {
    throw new ApiError(OFFLINE_MESSAGE, 0);
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(messageFromBody(body, response.status), response.status);
  }
  return body as T;
}

/** ファイルを送る（アップロードの進み具合を受け取れるよう XMLHttpRequest を使う）。 */
export function upload<T>(
  path: string,
  form: FormData,
  onProgress?: (fraction: number) => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}${path}`);
    xhr.setRequestHeader(CLIENT_HEADER, "1");
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as T);
      } else {
        reject(new ApiError(messageFromBody(xhr.response, xhr.status), xhr.status));
      }
    };
    xhr.onerror = () => reject(new ApiError(OFFLINE_MESSAGE, 0));
    xhr.send(form);
  });
}

export const api = {
  system: () => request<SystemInfo>("/api/system"),

  models: () =>
    request<{ models: VoiceModel[] }>("/api/models").then((r) => r.models),

  /** モデル1つの情報（学習用の音声の一覧つき）。 */
  model: (name: string) =>
    request<{ model: VoiceModel }>(`/api/models/${encodeURIComponent(name)}`).then(
      (r) => r.model,
    ),

  deleteModel: (name: string) =>
    request<{ ok: true }>(`/api/models/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),

  /** 学習の記録と、途中で保存したモデルの評価（試し聴き用の音声つき）。 */
  quality: (name: string) =>
    request<QualityInfo>(`/api/models/${encodeURIComponent(name)}/quality`),

  /** 途中で保存したモデルファイルを1つ消す。 */
  deleteWeight: (name: string, file: string) =>
    request<{ model: VoiceModel }>(
      `/api/models/${encodeURIComponent(name)}/weights/${encodeURIComponent(file)}`,
      { method: "DELETE" },
    ),

  /** 保存容量を空ける。 */
  cleanup: (name: string, options: { weights?: boolean; checkpoints?: boolean; stages?: boolean }) =>
    request<{ freed: number; storage: ModelStorage; model: VoiceModel }>(
      `/api/models/${encodeURIComponent(name)}/cleanup`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(options),
      },
    ),

  /** 取り込んだモデルの声の高さを設定する（声の高さを自動で合わせるのに使う）。 */
  setVoiceType: (name: string, voiceType: VoiceType | null) =>
    request<{ model: VoiceModel }>(`/api/models/${encodeURIComponent(name)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice_type: voiceType }),
    }),

  jobs: (kind?: Job["kind"], limit = 30) => {
    const query = new URLSearchParams({ limit: String(limit) });
    if (kind) query.set("kind", kind);
    return request<{ jobs: Job[] }>(`/api/jobs?${query}`).then((r) => r.jobs);
  },

  job: (id: string, logs = 120) =>
    request<{ job: Job }>(`/api/jobs/${id}?logs=${logs}`).then((r) => r.job),

  cancelJob: (id: string) =>
    request<{ job: Job }>(`/api/jobs/${id}/cancel`, { method: "POST" }),

  deleteJob: (id: string) =>
    request<{ ok: true }>(`/api/jobs/${id}`, { method: "DELETE" }),

  /** 保存済みのところから、合計エポック数まで学習を続ける。 */
  resumeTraining: (name: string, totalEpochs: number) => {
    const form = new FormData();
    form.set("name", name);
    form.set("resume", "true");
    form.set("total_epochs", String(totalEpochs));
    return request<{ job: TrainJob }>("/api/train", {
      method: "POST",
      body: form,
    }).then((r) => r.job);
  },
};

export const fileUrls = {
  output: (jobId: string, download = false) =>
    `${API_URL}/api/jobs/${jobId}/output${download ? "?download=1" : ""}`,
  stem: (jobId: string, stem: "vocals" | "instrumental", download = false) =>
    `${API_URL}/api/jobs/${jobId}/stems/${stem}${download ? "?download=1" : ""}`,
  input: (jobId: string) => `${API_URL}/api/jobs/${jobId}/input`,
  modelFile: (name: string, file: string) =>
    `${API_URL}/api/models/${encodeURIComponent(name)}/files/${encodeURIComponent(file)}`,
  datasetFile: (name: string, file: string) =>
    `${API_URL}/api/models/${encodeURIComponent(name)}/dataset/${encodeURIComponent(file)}`,
  datasetStage: (name: string, file: string, stage: DatasetStage) =>
    `${API_URL}/api/models/${encodeURIComponent(name)}/dataset/${encodeURIComponent(file)}/stages/${stage}`,
  sample: (name: string, epoch: number) =>
    `${API_URL}/api/models/${encodeURIComponent(name)}/samples/${epoch}`,
};

/**
 * 受け付ける音声ファイルの拡張子（サーバーの rvc_api/audio.py と合わせる）。
 * 動画ファイルは音声だけを取り出して使う。
 */
const AUDIO_EXTENSIONS = [
  ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff", ".caf", ".opus", ".wma", ".alac", ".webm",
];
const VIDEO_EXTENSIONS = [".mp4", ".mov", ".m4v", ".mkv", ".avi"];

export const AUDIO_ACCEPT = ["audio/*", "video/*", ...AUDIO_EXTENSIONS, ...VIDEO_EXTENSIONS].join(",");

export function isAudioFile(file: File): boolean {
  const name = file.name.toLowerCase();
  return [...AUDIO_EXTENSIONS, ...VIDEO_EXTENSIONS].some((ext) => name.endsWith(ext));
}

/** モデル名のルール（サーバーの rvc_api/library.py と合わせる） */
export const MODEL_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$/;

/** Windows でファイル名・フォルダ名に使えない名前（サーバーの library.WINDOWS_RESERVED と合わせる） */
const WINDOWS_RESERVED = /^(con|prn|aux|nul|com[0-9]|lpt[0-9])$/i;

/**
 * 新しいモデルの名前の誤り（なければ null）。
 * existing を渡すと、同じ名前（大文字・小文字の違いは同じとみなす）のモデルがあるかも確かめる。
 */
export function modelNameError(name: string, existing: VoiceModel[] = []): string | null {
  if (!name) return null;
  if (/[^\x00-\x7F]/.test(name)) {
    return "日本語などの全角文字は使えません。半角英数字で入力してください（例: my-voice）。";
  }
  if (!MODEL_NAME_RE.test(name)) {
    return "半角英数字で始まり、半角英数字・「-」・「_」だけの40文字以内にしてください。";
  }
  if (WINDOWS_RESERVED.test(name) || name.toLowerCase() === "reference") {
    return `「${name}」はシステムで使う名前のため使えません。別の名前にしてください。`;
  }
  const same = existing.find((model) => model.name.toLowerCase() === name.toLowerCase());
  // 学習が始まる前に止まった「未完成」のモデルは、同じ名前で作り直せる
  if (same && (same.weights.length > 0 || same.can_resume || same.active_job_id)) {
    return `「${same.name}」という名前のモデルはすでにあります。別の名前にするか、モデル一覧の「続きから学習」を使ってください。`;
  }
  return null;
}
