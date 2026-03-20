import glob
import os
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import runpod

COMFYUI_DIR = Path(os.getenv("COMFYUI_DIR", "/comfyui/ComfyUI"))
OUTPUT_DIR = Path(os.getenv("COMFYUI_OUTPUT_DIR", "/comfyui/output"))
MODELS_DIR = COMFYUI_DIR / "models"
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_API = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
COMFYUI_LOG_PATH = Path(os.getenv("COMFYUI_LOG_PATH", "/tmp/comfyui.log"))

COMFYUI_START_TIMEOUT_SECONDS = int(os.getenv("COMFYUI_START_TIMEOUT_SECONDS", "240"))
PROMPT_TIMEOUT_SECONDS = int(os.getenv("PROMPT_TIMEOUT_SECONDS", "7200"))
COMFYUI_HTTP_TIMEOUT_SECONDS = int(os.getenv("COMFYUI_HTTP_TIMEOUT_SECONDS", "30"))
TELEGRAM_HTTP_TIMEOUT_SECONDS = int(os.getenv("TELEGRAM_HTTP_TIMEOUT_SECONDS", "120"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "7200"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.5"))

TELEGRAM_NODE_ID = os.getenv("TELEGRAM_NODE_ID", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_INJECT_REQUIRED = os.getenv("TELEGRAM_INJECT_REQUIRED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
OUTPUT_SETTLE_TIMEOUT_SECONDS = int(os.getenv("OUTPUT_SETTLE_TIMEOUT_SECONDS", "120"))
OUTPUT_SETTLE_SECONDS = int(os.getenv("OUTPUT_SETTLE_SECONDS", "4"))

ALLOWED_MODEL_TYPES = {
    "checkpoints",
    "vae",
    "loras",
    "controlnet",
    "clip",
    "clip_vision",
    "unet",
    "diffusion_models",
    "text_encoders",
    "upscale_models",
}

_server_lock = threading.Lock()
_server_process: subprocess.Popen | None = None
_server_log_handle = None


class WorkerError(RuntimeError):
    pass


def _safe_model_name(name: str) -> str:
    safe_name = Path(name).name
    if safe_name != name or not safe_name:
        raise WorkerError(f"Invalid model name: {name!r}")
    return safe_name


def _safe_model_type(model_type: str) -> str:
    if model_type not in ALLOWED_MODEL_TYPES:
        raise WorkerError(f"Unknown model type: {model_type!r}")
    return model_type


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise WorkerError(f"URL must be http/https: {url!r}")
    return url


def _log_tail(max_lines: int = 120) -> str:
    if _server_log_handle is not None:
        _server_log_handle.flush()
    if not COMFYUI_LOG_PATH.exists():
        return ""
    with COMFYUI_LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:]).strip()


def _request_with_retry(
    method: str,
    url: str,
    *,
    max_attempts: int = 3,
    timeout: int = COMFYUI_HTTP_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.request(method=method, url=url, timeout=timeout, **kwargs)
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise WorkerError(f"Request failed after retries: {url} ({last_exc})")


def _is_comfyui_ready() -> bool:
    try:
        response = requests.get(f"{COMFYUI_API}/system_stats", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _start_comfyui() -> None:
    global _server_process
    global _server_log_handle

    with _server_lock:
        if _server_process and _server_process.poll() is None and _is_comfyui_ready():
            return

        if _server_process and _server_process.poll() is None:
            _server_process.terminate()
            try:
                _server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _server_process.kill()
                _server_process.wait(timeout=10)
            _server_process = None

        if _server_log_handle is not None:
            _server_log_handle.close()
            _server_log_handle = None

        COMFYUI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _server_log_handle = COMFYUI_LOG_PATH.open("w", encoding="utf-8")

        cmd = [
            "python",
            "main.py",
            "--listen",
            COMFYUI_HOST,
            "--port",
            str(COMFYUI_PORT),
            "--disable-auto-launch",
        ]
        _server_process = subprocess.Popen(
            cmd,
            cwd=str(COMFYUI_DIR),
            stdout=_server_log_handle,
            stderr=subprocess.STDOUT,
        )

        deadline = time.time() + COMFYUI_START_TIMEOUT_SECONDS
        while time.time() < deadline:
            if _server_process.poll() is not None:
                raise WorkerError(
                    "ComfyUI exited during cold start. "
                    f"Log tail from {COMFYUI_LOG_PATH}:\n{_log_tail()}"
                )
            if _is_comfyui_ready():
                return
            time.sleep(1)

        raise WorkerError(
            "ComfyUI cold start timed out. "
            f"Log tail from {COMFYUI_LOG_PATH}:\n{_log_tail()}"
        )


def _validate_workflow(workflow: Any) -> dict[str, Any]:
    if not isinstance(workflow, dict) or not workflow:
        raise WorkerError("input.workflow must be a non-empty ComfyUI API workflow object")
    return workflow


def _validate_models(models: Any) -> list[dict[str, Any]]:
    if models is None:
        return []
    if not isinstance(models, list):
        raise WorkerError("input.models must be a list")
    normalized: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            raise WorkerError("Each model entry must be an object")
        normalized.append(item)
    return normalized


def _download_model(model: dict[str, Any]) -> dict[str, Any]:
    model_type = _safe_model_type(str(model.get("type", "")))
    model_name = _safe_model_name(str(model.get("name", "")))
    model_url = _validate_url(str(model.get("url", "")))

    target_dir = MODELS_DIR / model_type
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / model_name

    if target_path.exists() and target_path.stat().st_size > 0:
        return {
            "type": model_type,
            "name": model_name,
            "path": str(target_path),
            "cached": True,
            "size_bytes": target_path.stat().st_size,
        }

    cmd = [
        "aria2c",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--continue=true",
        "--allow-overwrite=true",
        "--file-allocation=none",
        "--summary-interval=0",
        "--console-log-level=warn",
        "--timeout=30",
        "--max-tries=6",
        "--retry-wait=2",
        "--dir",
        str(target_dir),
        "--out",
        model_name,
        model_url,
    ]
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise WorkerError(
            f"aria2c failed for {model_name}: {result.stderr[-2000:] or result.stdout[-2000:]}"
        )

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise WorkerError(f"Downloaded file missing/empty: {target_path}")

    return {
        "type": model_type,
        "name": model_name,
        "path": str(target_path),
        "cached": False,
        "size_bytes": target_path.stat().st_size,
        "download_seconds": round(time.time() - started, 2),
    }


def _inject_telegram_if_present(workflow: dict[str, Any]) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    target_node_ids: list[str] = []
    if TELEGRAM_NODE_ID:
        if TELEGRAM_NODE_ID in workflow:
            target_node_ids = [TELEGRAM_NODE_ID]
        elif TELEGRAM_INJECT_REQUIRED:
            raise WorkerError(f"TELEGRAM_NODE_ID {TELEGRAM_NODE_ID!r} not found in workflow")
        else:
            return
    else:
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            if str(node.get("class_type", "")).strip() == "TelegramSender":
                target_node_ids.append(str(node_id))

        if not target_node_ids and TELEGRAM_INJECT_REQUIRED:
            raise WorkerError("No TelegramSender node found and TELEGRAM_INJECT_REQUIRED=true")

    for node_id in target_node_ids:
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
            node["inputs"] = inputs
        inputs["bot_token"] = TELEGRAM_BOT_TOKEN
        inputs["chat_id"] = TELEGRAM_CHAT_ID


def _submit_prompt(workflow: dict[str, Any]) -> str:
    payload = {"client_id": str(uuid.uuid4()), "prompt": workflow}
    response = _request_with_retry("POST", f"{COMFYUI_API}/prompt", json=payload, max_attempts=4)
    response.raise_for_status()
    data = response.json()

    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise WorkerError(f"ComfyUI did not return prompt_id: {data}")
    return str(prompt_id)


def _progress_percent() -> int | None:
    try:
        response = requests.get(f"{COMFYUI_API}/progress", timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        value = float(data.get("value", 0))
        maximum = float(data.get("max", 0))
        if maximum <= 0:
            return None
        pct = int((value / maximum) * 100)
        if pct < 0:
            return 0
        if pct > 100:
            return 100
        return pct
    except (requests.RequestException, ValueError, TypeError):
        return None


def _wait_for_prompt_completion(prompt_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    seen_progress_100 = False

    while time.time() < deadline:
        progress = _progress_percent()
        if progress is not None and progress >= 100:
            seen_progress_100 = True

        response = _request_with_retry(
            "GET",
            f"{COMFYUI_API}/history/{prompt_id}",
            max_attempts=3,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        if prompt_id in data:
            item = data[prompt_id]
            status = item.get("status", {}) if isinstance(item, dict) else {}
            status_str = str(status.get("status_str", "")).lower()

            if status_str in {"error", "failed"}:
                raise WorkerError(f"Prompt failed: {status.get('messages')}")

            if status_str in {"success", "completed"}:
                if not seen_progress_100:
                    # Ensure the generation reached 100% before upload.
                    time.sleep(1)
                    progress = _progress_percent()
                    if progress is not None and progress >= 100:
                        seen_progress_100 = True

                if not seen_progress_100:
                    raise WorkerError(
                        "Render completed but progress never reported 100%. "
                        "Aborting upload to satisfy 100% completion requirement."
                    )
                return item

        time.sleep(1)

    raise WorkerError(f"Prompt timeout reached for {prompt_id}")


def _newest_mp4_file() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = glob.glob(str(OUTPUT_DIR / "*.mp4"))
    if not candidates:
        raise WorkerError(f"No .mp4 files found in {OUTPUT_DIR}")
    latest = max(candidates, key=lambda p: Path(p).stat().st_mtime)
    return Path(latest)


def _wait_for_mp4_settle(file_path: Path, timeout_seconds: int, settle_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_size = -1
    stable_since: float | None = None

    while time.time() < deadline:
        if not file_path.exists():
            stable_since = None
            last_size = -1
            time.sleep(1)
            continue

        size = file_path.stat().st_size
        if size <= 0:
            stable_since = None
            last_size = size
            time.sleep(1)
            continue

        if size != last_size:
            last_size = size
            stable_since = time.time()
        elif stable_since is not None and (time.time() - stable_since) >= settle_seconds:
            return

        time.sleep(1)

    raise WorkerError(f"Output MP4 did not settle before timeout: {file_path}")


def _send_video_to_telegram(video_path: Path, caption: str | None = None) -> dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise WorkerError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required for Telegram upload")

    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    data = {"chat_id": TELEGRAM_CHAT_ID}
    if caption:
        data["caption"] = caption

    with video_path.open("rb") as video_file:
        files = {"video": (video_path.name, video_file, "video/mp4")}
        response = _request_with_retry(
            "POST",
            telegram_url,
            data=data,
            files=files,
            timeout=TELEGRAM_HTTP_TIMEOUT_SECONDS,
            max_attempts=3,
        )

    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise WorkerError(f"Telegram API returned error: {data}")
    return data


def handler(event: dict[str, Any]) -> dict[str, Any]:
    started = time.time()

    try:
        payload = event.get("input", {}) if isinstance(event, dict) else {}
        workflow = _validate_workflow(payload.get("workflow"))
        models = _validate_models(payload.get("models"))

        downloaded_models: list[dict[str, Any]] = []
        for model in models:
            downloaded_models.append(_download_model(model))

        _start_comfyui()

        _inject_telegram_if_present(workflow)
        prompt_id = _submit_prompt(workflow)
        history = _wait_for_prompt_completion(
            prompt_id=prompt_id,
            timeout_seconds=int(payload.get("prompt_timeout_seconds", PROMPT_TIMEOUT_SECONDS)),
        )

        newest_mp4 = _newest_mp4_file()
        _wait_for_mp4_settle(
            file_path=newest_mp4,
            timeout_seconds=OUTPUT_SETTLE_TIMEOUT_SECONDS,
            settle_seconds=OUTPUT_SETTLE_SECONDS,
        )
        telegram_result = _send_video_to_telegram(
            video_path=newest_mp4,
            caption=str(payload.get("telegram_caption", "RunPod render completed")),
        )

        return {
            "status": "success",
            "prompt_id": prompt_id,
            "mp4_path": str(newest_mp4),
            "downloaded_models": downloaded_models,
            "history_summary_keys": list(history.keys()) if isinstance(history, dict) else [],
            "telegram_message_id": telegram_result.get("result", {}).get("message_id"),
            "duration_seconds": round(time.time() - started, 2),
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
            "comfyui_log_tail": _log_tail(),
            "duration_seconds": round(time.time() - started, 2),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
