"""RVC Studio の API サーバー（FastAPI）。

起動方法: プロジェクトのフォルダで `npm run rvc:server`（または画面とまとめて `npm run app`）
API の一覧: サーバー起動後に http://127.0.0.1:8765/docs
"""

import os
import platform
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import audio, evaluation, library, pipelines, pretrains, separation, settings, system
from . import sources as url_sources
from .jobs import ACTIVE, JobManager

jobs = JobManager()

# ピッチの検出方法。Applio 3.7.1 の hybrid[...] は CLI の選択肢にあるが中身が作られておらず、
# 必ず失敗する（rvc/infer/pipeline.py の get_f0）ため、受け付けない
F0_METHODS = ("rmvpe", "fcpe", "crepe")
INPUT_KINDS = ("voice", "song")
HARMONY_MODES = ("together", "convert", "keep", "remove")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    library.ensure_reference_files()
    jobs.register("train", pipelines.run_training)
    jobs.register(
        "convert", pipelines.run_conversion, cleanup=pipelines.cleanup_conversion
    )
    jobs.start()
    # 履歴から消えた変換結果や、前回落ちたときの作業用ファイルを片付ける
    freed = pipelines.sweep_orphan_files(jobs.list_jobs(limit=10**9))
    if freed:
        size = f"{freed / 1e6:.0f}MB" if freed >= 1e6 else f"{max(freed / 1e3, 1):.0f}KB"
        print(f"[RVC Studio] 使われていない変換結果・作業用ファイル（{size}）を片付けました。", flush=True)
    threading.Thread(target=system.detect, daemon=True).start()
    yield
    jobs.shutdown()


app = FastAPI(title="RVC Studio API", lifespan=lifespan)

# データを変える操作には、画面（Next.js）が付ける専用のヘッダーを必須にする。
# ブラウザは別のサイトから専用ヘッダー付きのリクエストを送る前に CORS の確認をするため、
# ほかのサイトを開いたときに、勝手に学習やダウンロードを始められることを防げる。
CLIENT_HEADER = "x-rvc-client"
# この PC で動いている画面（Next.js）のアドレス
LOCAL_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


def _allowed_hosts() -> set[str]:
    """受け付けるアドレス（Host ヘッダー）。この PC を指す名前と、起動時に指定したアドレス。"""
    hosts = set(LOCAL_HOSTS)
    if settings.HOST not in ("0.0.0.0", "::"):
        hosts.add(settings.HOST.lower())
    # 別の名前で使いたいときは、環境変数でカンマ区切りで足せる（例: RVC_ALLOWED_HOSTS=my-mac.local）
    hosts |= {item.strip().lower() for item in os.environ.get("RVC_ALLOWED_HOSTS", "").split(",") if item.strip()}
    return hosts


def _host_name(host_header: str) -> str:
    """Host ヘッダーから、ポート番号を除いた名前を取り出す（"[::1]:8765" → "[::1]"）。"""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host.split("]", 1)[0] + "]"
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


ALLOWED_HOSTS = _allowed_hosts()


@app.middleware("http")
async def _protect_local_server(request: Request, call_next):
    # 悪意のあるサイトが自分のドメインの行き先をこの PC にすり替える手口（DNS リバインディング）では、
    # ブラウザからは「同じサイト」に見えるため CORS も専用ヘッダーの確認も効かない。
    # そのため、この PC を指す名前で呼ばれたときだけ応じる
    if _host_name(request.headers.get("host", "")) not in ALLOWED_HOSTS:
        return JSONResponse(status_code=403, content={"detail": "このアドレスからのアクセスは受け付けていません。"})
    origin = request.headers.get("origin")
    if origin is not None and not LOCAL_ORIGIN_RE.match(origin):
        return JSONResponse(status_code=403, content={"detail": "このアプリの画面以外からの操作は受け付けていません。"})
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and not request.headers.get(
        CLIENT_HEADER
    ):
        return JSONResponse(
            status_code=403,
            content={"detail": "このアプリの画面以外からの操作は受け付けていません。"},
        )
    return await call_next(request)


# 同じ PC で動いている画面（Next.js）からのアクセスだけ許可する
# （後から追加したものが外側で動くので、CORS は上の確認より先に処理される）
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LOCAL_ORIGIN_RE.pattern,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(library.LibraryError)
@app.exception_handler(audio.AudioError)
@app.exception_handler(url_sources.SourceError)
@app.exception_handler(separation.PrepError)
@app.exception_handler(pretrains.PretrainedError)
async def _user_error(_request: Request, error: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


def _missing_parts() -> list[str]:
    missing = []
    if not (settings.APPLIO_DIR / "core.py").is_file():
        missing.append("Applio 本体")
    if not settings.PREDICTOR_RMVPE.is_file():
        missing.append("ピッチ検出モデル（rmvpe）")
    if not settings.EMBEDDER_CONTENTVEC.is_file():
        missing.append("声の特徴を取り出すモデル（contentvec）")
    return missing


def _require_ready() -> None:
    missing = _missing_parts()
    if missing:
        raise HTTPException(
            503,
            "セットアップが終わっていません（不足: "
            + "、".join(missing)
            + "）。`npm run rvc:setup` を実行してください。",
        )


def _check_range(label: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise HTTPException(400, f"{label}は {low}〜{high} の範囲で指定してください。")


def _save_upload(upload: UploadFile, target: Path) -> None:
    with target.open("wb") as out:
        shutil.copyfileobj(upload.file, out, length=1024 * 1024)
    if target.stat().st_size == 0:
        raise audio.AudioError(f"「{upload.filename}」が空のファイルです。")


# ------------------------------------------------------------------ 状態


@app.get("/")
def root() -> dict:
    return {"name": "RVC Studio API", "docs": "/docs"}


@app.get("/api/system")
def get_system() -> dict:
    missing = _missing_parts()
    usage = shutil.disk_usage(settings.DATA_DIR)
    device = system.info()
    return {
        "ok": True,
        "ready": not missing,
        "missing": missing,
        "applio_version": settings.applio_version(),
        "platform": {"windows": "windows", "darwin": "mac"}.get(platform.system().lower(), "linux"),
        "device": device.get("device") or "checking",
        "device_name": device.get("name"),
        "device_memory_gb": device.get("memory_gb"),
        "recommended_batch_size": system.recommended_batch_size(),
        "cpu_count": os.cpu_count(),
        "disk_free_bytes": usage.free,
        "ffmpeg": settings.ffmpeg_path() is not None,
        "accepted_extensions": sorted(audio.accepted_extensions()),
        "pretrained": {
            str(rate): pretrains.is_downloaded(pretrains.STANDARD, rate)
            for rate in settings.SAMPLE_RATES
        },
        # 学習の出発点にできる事前学習モデル（RVC 標準とコミュニティの改良版）
        "pretrains": pretrains.catalog(),
        "embedders": list(library.EMBEDDERS),
        # BGM・伴奏を取り除く機能（ボーカル分離）が使えるか、AI モデルをダウンロード済みか
        "separation": separation.status(),
        "active_jobs": [
            job.to_dict(log_lines=0)
            for job in jobs.list_jobs(limit=1000)
            if job.status in ACTIVE
        ],
    }


# ------------------------------------------------------------------ モデル


@app.get("/api/models")
def get_models() -> dict:
    models = library.list_models()
    for model in models:
        job = jobs.active_for_model(model["name"])
        model["active_job_id"] = job.id if job else None
        model["active_job_status"] = job.status if job else None
    return {"models": models}


@app.get("/api/models/{name}")
def get_model(name: str) -> dict:
    name = library.resolve_model(name)
    model = library.model_summary(name)
    job = jobs.active_for_model(name)
    model["active_job_id"] = job.id if job else None
    model["active_job_status"] = job.status if job else None
    model["dataset"] = library.dataset_files(name)
    model["storage"] = library.storage(name)
    return {"model": model}


@app.get("/api/models/{name}/quality")
def get_model_quality(name: str) -> dict:
    """学習の記録（エポックごとの誤差）と、途中で保存したモデルの評価結果（試し聴き用の音声つき）。"""
    name = library.resolve_model(name)
    directory = library.model_dir(name)
    weights = library.list_weights(name)
    available = {w["epoch"] for w in weights if w["epoch"] is not None}
    items = evaluation.read_evaluations(directory)
    recommended = library.recommended_weight(name, weights)
    meta = library.read_meta(name)
    return {
        "curve": evaluation.training_curve(directory, library.steps_per_epoch(weights)),
        "evaluations": [
            {
                "epoch": item["epoch"],
                "val_loss": item.get("val_loss"),
                "has_sample": bool(item.get("sample")) and evaluation.sample_path(directory, item["epoch"]).is_file(),
                "available": item["epoch"] in available,
                "weight": next((w["file"] for w in weights if w["epoch"] == item["epoch"]), None),
            }
            for item in items
        ],
        "recommended_epoch": recommended["epoch"] if recommended else None,
        "validation_clips": meta.get("validation_clips"),
        "early_stopped_at": meta.get("early_stopped_at"),
    }


@app.get("/api/models/{name}/samples/{epoch}")
def get_model_sample(name: str, epoch: int) -> FileResponse:
    name = library.resolve_model(name)
    return FileResponse(library.sample_file(name, epoch), media_type="audio/flac", content_disposition_type="inline")


@app.delete("/api/models/{name}/weights/{filename}")
def delete_model_weight(name: str, filename: str) -> dict:
    name = library.resolve_model(name)
    if jobs.uses_model(name):
        raise HTTPException(409, "このモデルは学習中、または変換待ちのため、ファイルを削除できません。")
    library.delete_weight(name, filename)
    return {"model": library.model_summary(name)}


@app.post("/api/models/{name}/cleanup")
def cleanup_model(
    name: str,
    weights: bool = Body(False),
    checkpoints: bool = Body(False),
    stages: bool = Body(False),
) -> dict:
    """保存容量を空ける（いらないエポック・続きから学習するためのファイル・下ごしらえの途中経過）。"""
    name = library.resolve_model(name)
    if jobs.uses_model(name):
        raise HTTPException(409, "このモデルは学習中、または変換待ちのため、整理できません。")
    result = library.cleanup(name, weights=weights, checkpoints=checkpoints, stages=stages)
    return {**result, "storage": library.storage(name), "model": library.model_summary(name)}


@app.patch("/api/models/{name}")
def update_model(name: str, voice_type: str | None = Body(None, embed=True)) -> dict:
    """取り込んだモデルの「声の高さ」を設定する（声の高さを自動で合わせるときに使う）。"""
    name = library.resolve_model(name)
    if voice_type not in (None, "", *library.VOICE_TYPES):
        raise HTTPException(400, "声の高さの指定が正しくありません。")
    library.write_meta(name, voice_type=voice_type or None)
    return {"model": library.model_summary(name)}


@app.delete("/api/models/{name}")
def delete_model(name: str) -> dict:
    name = library.resolve_model(name)
    if jobs.uses_model(name):
        raise HTTPException(409, "このモデルは学習中、または変換待ちのため削除できません。")
    library.delete_model(name)
    return {"ok": True}


@app.post("/api/models/import")
def import_model(
    name: str = Form(...),
    model_file: UploadFile = File(...),
    index_file: UploadFile | None = File(None),
    voice_type: str | None = Form(None),
    embedder: str | None = Form(None),
) -> dict:
    if index_file is not None and not index_file.filename:
        index_file = None
    return {
        "model": library.import_model(
            name, model_file, index_file, voice_type=voice_type or None, embedder=embedder or None
        )
    }


@app.get("/api/models/{name}/files/{filename}")
def download_model_file(name: str, filename: str) -> FileResponse:
    name = library.resolve_model(name)
    path = library.file_in_model(name, filename)
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.get("/api/models/{name}/dataset/{filename}")
def get_dataset_file(name: str, filename: str) -> FileResponse:
    """学習用の音声を聴くためのもの（BGM・伴奏を取り除いた結果の確認など）。"""
    name = library.resolve_model(name)
    path = library.dataset_file(name, filename)
    return FileResponse(path, content_disposition_type="inline")


@app.get("/api/models/{name}/dataset/{filename}/stages/{stage}")
def get_dataset_stage(name: str, filename: str, stage: str) -> FileResponse:
    """下ごしらえの途中経過（original = 下ごしらえの前、vocals = BGM 除去の後 など）。"""
    name = library.resolve_model(name)
    path = library.stage_file(name, filename, stage)
    return FileResponse(path, media_type="audio/flac", content_disposition_type="inline")


# ------------------------------------------------------------------ 学習


@app.post("/api/train")
def start_training(
    name: str = Form(...),
    sample_rate: int = Form(40000),
    total_epochs: int = Form(200),
    batch_size: int | None = Form(None),
    save_every: int | None = Form(None),
    resume: bool = Form(False),
    # 声の特徴を取り出すモデル（contentvec など）と、学習の出発点にする事前学習モデル
    embedder: str | None = Form(None),
    pretrained: str | None = Form(None),
    # 学習しすぎを見分けて自動で止めるか
    early_stop: bool | None = Form(None),
    # BGM・伴奏の除去の精度（high: 2つの AI の組み合わせ / fast: 速さ優先）
    separation_quality: str | None = Form(None),
    files: list[UploadFile] | None = File(None),
    # 動画の URL の一覧（JSON: [{"url": "...", "start": "1:00", "end": "5:30"}, ...]）
    sources: str | None = Form(None),
    # 学習の前に行う音声の下ごしらえ（例: "vocals,dereverb"。rvc_api/separation.py を参照）
    prep: str | None = Form(None),
    # 「自分の声、または本人の許可がある」ことの確認
    consent: bool = Form(False),
) -> dict:
    _require_ready()
    uploads = [upload for upload in (files or []) if upload.filename]
    video_sources = url_sources.parse_sources(sources)
    # 続きから学習するときは、音声をもう一度処理しない
    prep_steps = [] if resume else separation.parse_steps(prep)
    if resume:
        name = library.resolve_model(name)
    # 続きから学習するときは、指定がなければ最初の学習と同じ設定を使う
    # （メモリ不足でバッチサイズを下げていた場合などに、元の値へ戻らないように）
    meta = library.read_meta(name) if resume else {}
    batch_size = batch_size or int(meta.get("batch_size") or system.recommended_batch_size())
    save_every = save_every or int(meta.get("save_every") or 10)
    embedder = library.validate_embedder(embedder or meta.get("embedder"))
    early_stop = bool(meta.get("early_stop", True)) if early_stop is None else early_stop
    quality = separation.parse_quality(separation_quality)
    _check_range("エポック数", total_epochs, 1, 10000)
    _check_range("バッチサイズ", batch_size, 1, 32)
    _check_range("保存の間隔", save_every, 1, 100)

    if resume:
        if jobs.active_for_model(name):
            raise HTTPException(409, "このモデルはすでに学習中（または学習待ち）です。")
        model = library.model_summary(name)
        if model["checkpoint_broken"]:
            raise HTTPException(
                400,
                "再開用のファイルが壊れているため、続きから学習できません（前回、保存の途中で止まった可能性があります）。"
                "今あるモデルはそのまま変換に使えます。",
            )
        if not model["can_resume"]:
            raise HTTPException(
                400, "このモデルは続きから学習できません（再開用のデータがありません）。"
            )
        if uploads or video_sources:
            raise HTTPException(400, "続きから学習するときは、音声を追加できません。")
        start_epoch = library.checkpoint_epoch(name)
        if total_epochs <= start_epoch:
            raise HTTPException(
                400, f"合計エポック数は、いまの {start_epoch} より大きくしてください。"
            )
        sample_rate = int(model["sample_rate"] or sample_rate)
        pretrained = pretrains.validate(meta.get("pretrained"), sample_rate)
        dataset = library.dataset_files(name)
    else:
        if sample_rate not in settings.SAMPLE_RATES:
            raise HTTPException(400, "サンプルレートは 32000・40000・48000 から選んでください。")
        pretrained = pretrains.validate(pretrained, sample_rate)
        name = library.validate_name(name)
        if jobs.active_for_model(name):
            raise HTTPException(409, "このモデルはすでに学習中（または学習待ち）です。")
        if library.model_dir(name).exists():
            existing = library.model_summary(name)
            if existing and not existing["weights"] and not existing["can_resume"]:
                # 学習が始まる前に失敗した「未完成」のモデルは、同じ名前で作り直せるようにする
                library.delete_model(name)
            else:
                raise HTTPException(
                    409,
                    f"「{name}」という名前のモデルはすでにあります。別の名前にするか、"
                    "モデル一覧の「続きから学習」を使ってください。",
                )
        if not uploads and not video_sources:
            raise HTTPException(400, "学習用の音声ファイルを選ぶか、動画の URL を入力してください。")
        if video_sources and not consent:
            raise HTTPException(
                400, "動画の声が自分の声か、本人の許可があることを確認して、チェックを入れてください。"
            )
        # 前回の失敗などで残った音声があれば消してから保存する
        shutil.rmtree(library.dataset_dir(name), ignore_errors=True)
        try:
            dataset = library.save_dataset_files(name, uploads) if uploads else []
        except Exception:
            shutil.rmtree(library.dataset_dir(name), ignore_errors=True)
            raise
        silent = [path.name for path in library.dataset_paths(name) if audio.is_silent(path)]
        if silent:
            shutil.rmtree(library.dataset_dir(name), ignore_errors=True)
            raise HTTPException(
                400,
                f"「{'」「'.join(silent)}」には声が入っていません（無音です）。声の入った音声を選んでください。",
            )
        total = sum(item["duration"] or 0 for item in dataset)
        # 動画の音声は学習の最初にダウンロードするので、長さの確認はそのときに行う
        if total < 10 and not video_sources:
            shutil.rmtree(library.dataset_dir(name), ignore_errors=True)
            raise HTTPException(
                400, "学習用の音声が短すぎます。合計10秒以上が必要です（10分以上がおすすめです）。"
            )
        library.model_dir(name).mkdir(parents=True)
        library.write_meta(
            name,
            source="trained",
            created_at=time.time(),
            sample_rate=sample_rate,
            batch_size=batch_size,
            save_every=save_every,
            prep=prep_steps,
            embedder=embedder,
            pretrained=pretrained,
            early_stop=early_stop,
            separation_quality=quality,
        )

    job = jobs.submit(
        "train",
        {
            "name": name,
            "sample_rate": sample_rate,
            "total_epochs": total_epochs,
            "batch_size": batch_size,
            "save_every": min(save_every, total_epochs),
            "resume": resume,
            "dataset_files": len(dataset),
            "dataset_duration": round(sum(item["duration"] or 0 for item in dataset), 1),
            "sources": video_sources,
            "prep": prep_steps,
            "embedder": embedder,
            "pretrained": pretrained,
            "early_stop": early_stop,
            "separation_quality": quality,
        },
    )
    if resume:
        library.write_meta(name, early_stop=early_stop)
    return {"job": job.to_dict()}


# ------------------------------------------------------------------ 変換


@app.post("/api/convert")
def start_conversion(
    model: str = Form(...),
    file: UploadFile = File(...),
    weight: str | None = Form(None),
    pitch: int = Form(0),
    # manual: pitch をそのまま使う / auto: 元の声とモデルの声の高さから自動で決める
    pitch_mode: str = Form("manual"),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.75),
    # 子音・息の保護。RVC では小さいほど強く保護し、0.5 で保護しない（Applio の初期値 0.33）
    protect: float = Form(0.33),
    volume_envelope: float = Form(1.0),
    autotune: bool = Form(False),
    autotune_strength: float = Form(1.0),
    formant: bool = Form(False),
    formant_quefrency: float = Form(1.0),
    formant_timbre: float = Form(1.0),
    clean_input: bool = Form(False),
    clean_output: bool = Form(False),
    clean_strength: float = Form(12),
    export_format: str = Form("wav"),
    # voice: 声だけの音声 / song: 伴奏の入った曲（声と伴奏を分けて、声だけ変換して合わせ直す）
    input_kind: str = Form("voice"),
    separation_quality: str | None = Form(None),
    dereverb: bool = Form(True),
    harmony: str = Form("together"),
    reverb: float = Form(0.3),
    vocal_gain_db: float = Form(0.0),
    instrumental_gain_db: float = Form(0.0),
) -> dict:
    _require_ready()
    name = library.resolve_model(model)
    weights = [w["file"] for w in library.list_weights(name)]
    if not weights:
        raise HTTPException(400, "このモデルはまだ使えません（学習が終わっていません）。")
    # 指定がなければ、学習しすぎていない「おすすめ」のエポックを使う
    weight = weight or (library.recommended_weight(name) or {}).get("file") or weights[-1]
    if weight not in weights:
        raise HTTPException(400, "指定されたモデルファイルが見つかりません。")
    _check_range("ピッチ", pitch, -24, 24)
    _check_range("インデックスの効き具合", index_rate, 0, 1)
    _check_range("子音の保護", protect, 0, 0.5)
    _check_range("音量の追従", volume_envelope, 0, 1)
    _check_range("オートチューンの強さ", autotune_strength, 0, 1)
    _check_range("フォルマント（声の響き）", formant_quefrency, 0.1, 4)
    _check_range("フォルマント（声色）", formant_timbre, 0.1, 4)
    _check_range("雑音を減らす強さ", clean_strength, 1, 30)
    _check_range("残響", reverb, 0, 1)
    _check_range("声の大きさ", vocal_gain_db, -12, 12)
    _check_range("伴奏の大きさ", instrumental_gain_db, -12, 12)
    if f0_method not in F0_METHODS:
        raise HTTPException(400, "ピッチの検出方法が正しくありません。")
    if pitch_mode not in ("manual", "auto"):
        raise HTTPException(400, "声の高さの決め方が正しくありません。")
    if pitch_mode == "auto" and library.target_pitch(name) is None:
        raise HTTPException(
            400,
            "このモデルは声の高さがわからないため、自動で合わせられません。"
            "モデル一覧で「声の高さ」を設定するか、ピッチを手動で指定してください。",
        )
    if export_format not in audio.EXPORT_FORMATS:
        raise HTTPException(400, "書き出す形式は WAV・FLAC・MP3 から選んでください。")
    if input_kind not in INPUT_KINDS:
        raise HTTPException(400, "音声の種類の指定が正しくありません。")
    if harmony not in HARMONY_MODES:
        raise HTTPException(400, "ハモリの扱いの指定が正しくありません。")
    quality = separation.parse_quality(separation_quality)
    if input_kind == "song":
        separation.parse_song_steps(dereverb, harmony != "together")

    original_name = file.filename or "audio.wav"
    audio.check_extension(original_name)
    upload_id = uuid.uuid4().hex[:12]
    work_dir = settings.UPLOADS_DIR / upload_id
    work_dir.mkdir(parents=True)
    try:
        target = work_dir / f"input{Path(original_name).suffix.lower()}"
        _save_upload(file, target)
        # 曲は、声と伴奏を分けるときに左右の違いも手がかりにするので、ステレオのまま残す
        target = audio.make_readable(target, mono=input_kind != "song", display_name=original_name)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    job = jobs.submit(
        "convert",
        {
            "model": name,
            "weight": weight,
            "pitch": pitch,
            "pitch_mode": pitch_mode,
            "f0_method": f0_method,
            "index_rate": index_rate,
            "protect": protect,
            "volume_envelope": volume_envelope,
            "autotune": autotune,
            "autotune_strength": autotune_strength,
            "formant": formant,
            "formant_quefrency": formant_quefrency,
            "formant_timbre": formant_timbre,
            "clean_input": clean_input,
            "clean_output": clean_output,
            "clean_strength": clean_strength,
            "export_format": export_format,
            "input_kind": input_kind,
            "separation_quality": quality,
            "dereverb": dereverb,
            "harmony": harmony,
            "reverb": reverb,
            "vocal_gain_db": vocal_gain_db,
            "instrumental_gain_db": instrumental_gain_db,
            "input_path": str(target),
            "input_name": original_name,
            "input_duration": audio.duration_of(target),
            "upload_id": upload_id,
        },
    )
    return {"job": job.to_dict()}


# ------------------------------------------------------------------ ジョブ


def _get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "ジョブが見つかりません。")
    return job


@app.get("/api/jobs")
def list_jobs(kind: str | None = None, limit: int = 30) -> dict:
    return {
        "jobs": [job.to_dict(log_lines=0) for job in jobs.list_jobs(kind, min(limit, 200))]
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, logs: int = 120) -> dict:
    return {"job": _get_job(job_id).to_dict(log_lines=min(max(logs, 0), 500))}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    _get_job(job_id)
    return {"job": jobs.cancel(job_id).to_dict(log_lines=0)}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = _get_job(job_id)
    if job.status in ACTIVE:
        raise HTTPException(409, "実行中のジョブは削除できません。先にキャンセルしてください。")
    jobs.delete(job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/output")
def get_job_output(job_id: str, download: bool = False) -> FileResponse:
    job = _get_job(job_id)
    if job.kind != "convert" or job.status != "done" or not job.result:
        raise HTTPException(404, "変換結果がありません。")
    return _output_file(job, job.result["output"], "", download)


@app.get("/api/jobs/{job_id}/stems/{stem}")
def get_job_stem(job_id: str, stem: str, download: bool = False) -> FileResponse:
    """曲の変換で作った「変換した声だけ」「伴奏だけ」の音声。"""
    job = _get_job(job_id)
    stems = (job.result or {}).get("stems") or {}
    if job.kind != "convert" or job.status != "done" or stem not in stems:
        raise HTTPException(404, "音声がありません。")
    label = {"vocals": "_声", "instrumental": "_伴奏"}.get(stem, f"_{stem}")
    return _output_file(job, stems[stem], label, download)


def _output_file(job, filename: str, label: str, download: bool) -> FileResponse:
    path = settings.OUTPUTS_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "変換結果のファイルが見つかりません。")
    fmt = path.suffix.lstrip(".").lower()
    media_type = audio.EXPORT_FORMATS.get(fmt, {}).get("media_type", "application/octet-stream")
    stem = Path(job.params.get("input_name") or "audio").stem
    return FileResponse(
        path,
        media_type=media_type,
        filename=f"{stem}_{job.params['model']}{label}.{fmt}",
        content_disposition_type="attachment" if download else "inline",
    )


@app.get("/api/jobs/{job_id}/input")
def get_job_input(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    path = Path(job.params.get("input_path") or "")
    if job.kind != "convert" or not path.is_file():
        raise HTTPException(404, "元の音声が見つかりません。")
    return FileResponse(path, content_disposition_type="inline")
