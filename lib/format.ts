import type { JobStatus, PrepStep } from "./api";

/** 秒数を「1時間5分」「3分20秒」「45秒」のように表す。 */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return m > 0 ? `${h}時間${m}分` : `${h}時間`;
  if (m > 0) return s > 0 && m < 10 ? `${m}分${s}秒` : `${m}分`;
  return `${s}秒`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  const digits = value >= 100 || unit === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

const dateFormatter = new Intl.DateTimeFormat("ja-JP", {
  month: "numeric",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

/** UNIX 時刻（秒）を「11/10 14:32」のように表す。 */
export function formatDate(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "—";
  return dateFormatter.format(new Date(epochSeconds * 1000));
}

export function formatPitch(semitones: number): string {
  if (semitones > 0) return `+${semitones}`;
  return String(semitones);
}

export function formatSampleRate(rate: number | null | undefined): string {
  return rate ? `${rate / 1000}kHz` : "—";
}

const PREP_NAMES: Record<PrepStep, string> = {
  vocals: "BGM除去",
  dereverb: "残響除去",
  karaoke: "ハモリ除去",
  denoise: "雑音除去",
};

/** 音声の下ごしらえを「BGM除去・残響除去」のように表す（何もしないなら空文字）。 */
export function formatPrep(steps: readonly PrepStep[] | null | undefined): string {
  return (steps ?? [])
    .map((step) => PREP_NAMES[step])
    .filter(Boolean)
    .join("・");
}

export const STATUS_LABELS: Record<JobStatus, string> = {
  queued: "順番待ち",
  running: "実行中",
  done: "完了",
  error: "失敗",
  cancelled: "キャンセル",
  interrupted: "中断",
};

/** 事前学習モデルの表示名（サーバーの rvc_api/pretrains.py と合わせる） */
export const PRETRAINED_NAMES: Record<string, string> = {
  standard: "RVC 標準",
  titan: "TITAN",
  snowie: "Snowie V3.1",
  klm4: "KLM 4",
  ov2super: "Ov2 Super",
};

export function formatPretrained(id: string | null | undefined): string | null {
  if (!id) return null;
  return PRETRAINED_NAMES[id] ?? id;
}

/** 「+3.0dB」「-6.0dB」のように表す。 */
export function formatDb(value: number): string {
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}dB`;
}

/** 声の高さ（Hz）を「約180Hz」のように表す。 */
export function formatHz(value: number | null | undefined): string {
  return value ? `約${Math.round(value)}Hz` : "—";
}
