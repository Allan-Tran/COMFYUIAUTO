import os
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
import runpod

COMFYUI_DIR = Path(os.getenv("COMFYUI_DIR", "/comfyui/ComfyUI"))
MODELS_DIR = COMFYUI_DIR / "models"
OUTPUT_DIR = COMFYUI_DIR / "output"
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_API = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "7200"))
PROMPT_TIMEOUT_SECONDS = int(os.getenv("PROMPT_TIMEOUT_SECONDS", "7200"))
DOWNLOAD_WORKERS = int(os.getenv("DOWNLOAD_WORKERS", "4"))
COMFYUI_LOG_PATH = Path(os.getenv("COMFYUI_LOG_PATH", "/tmp/comfyui.log"))

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
    if safe_name != name:
        raise WorkerError(f"Invalid model name path: {name}")
    if not safe_name:
        raise WorkerError("Model name is empty")
    return safe_name


def _safe_model_type(model_type: str) -> str:
    if model_type not in ALLOWED_MODEL_TYPES:
        raise WorkerError(f"Unknown model type: {model_type!r}")
    return model_type


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise WorkerError(f"URL scheme must be http or https: {url!r}")
    return url


def _log_tail(max_lines: int = 120) -> str:
    if _server_log_handle is not None:
        _server_log_handle.flush()

    if not COMFYUI_LOG_PATH.exists():
        return ""

    with COMFYUI_LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:]).strip()


def _download_model(model: Dict[str, Any]) -> Dict[str, Any]:
    model_type = model.get("type")
    url = model.get("url")
    model_name = model.get("name")

    if not model_type or not url or not model_name:
        raise WorkerError("Each model entry requires type, url, and name")

    safe_model_type = _safe_model_type(str(model_type))
    safe_model_name = _safe_model_name(str(model_name))
    safe_url = _validate_url(str(url))
    target_dir = MODELS_DIR / safe_model_type
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_model_name

    if target_path.exists() and target_path.stat().st_size > 0:
        return {
            "type": safe_model_type,
            "name": safe_model_name,
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
        safe_model_name,
        safe_url,
    ]

    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise WorkerError(
            f"aria2c failed for {safe_model_name}: {result.stderr[-2000:] or result.stdout[-2000:]}"
        )

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise WorkerError(f"Download finished but file missing/empty: {target_path}")

    return {
        "type": safe_model_type,
        "name": safe_model_name,
        "path": str(target_path),
        "cached": False,
        "size_bytes": target_path.stat().st_size,
        "download_seconds": round(time.time() - started, 2),
    }


def _is_comfyui_ready() -> bool:
    try:
        response = requests.get(f"{COMFYUI_API}/system_stats", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _start_comfyui() -> None:
    global _server_log_handle
    global _server_process

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

        deadline = time.time() + 180
        while time.time() < deadline:
            if _server_process.poll() is not None:
                tail = _log_tail()
                raise WorkerError(
                    "ComfyUI exited before becoming ready. "
                    f"Log tail from {COMFYUI_LOG_PATH}:\n{tail}"
                )
            if _is_comfyui_ready():
                return
            time.sleep(1)

        tail = _log_tail()
        raise WorkerError(
            "Timed out waiting for ComfyUI to become ready. "
            f"Log tail from {COMFYUI_LOG_PATH}:\n{tail}"
        )


def _latest_output_mtime() -> float:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = 0.0
    for file_path in OUTPUT_DIR.rglob("*"):
        if file_path.is_file():
            latest = max(latest, file_path.stat().st_mtime)
    return latest


def _submit_prompt(workflow: Dict[str, Any]) -> str:
    payload = {
        "client_id": str(uuid.uuid4()),
        "prompt": workflow,
    }
    response = requests.post(f"{COMFYUI_API}/prompt", json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise WorkerError(f"ComfyUI did not return prompt_id: {data}")

    return prompt_id


def _wait_for_prompt(prompt_id: str, timeout_seconds: int) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        response = requests.get(f"{COMFYUI_API}/history/{prompt_id}", timeout=20)
        response.raise_for_status()
        data = response.json()

        if prompt_id in data:
            item = data[prompt_id]
            status = item.get("status", {})
            status_str = status.get("status_str", "")

            if status_str in {"error", "failed"}:
                messages = status.get("messages")
                raise WorkerError(f"Prompt failed: {messages}")

            if status_str in {"success", "completed"}:
                return item

        time.sleep(1)

    raise WorkerError(f"Prompt timeout reached for {prompt_id}")


def _wait_for_output_settle(previous_mtime: float, timeout_seconds: int = 120, settle_seconds: int = 4) -> None:
    deadline = time.time() + timeout_seconds
    stable_since = None
    last_mtime = previous_mtime

    while time.time() < deadline:
        current_mtime = _latest_output_mtime()
        if current_mtime > last_mtime:
            last_mtime = current_mtime
            stable_since = time.time()
        elif stable_since is None:
            stable_since = time.time()
        elif time.time() - stable_since >= settle_seconds:
            return

        time.sleep(1)

    raise WorkerError("Output did not settle before timeout")


def _validate_workflow(workflow: Any) -> Dict[str, Any]:
    if not isinstance(workflow, dict) or not workflow:
        raise WorkerError("input.workflow must be a non-empty ComfyUI API workflow object")
    return workflow


def _validate_models(models: Any) -> List[Dict[str, Any]]:
    if models is None:
        return []
    if not isinstance(models, list):
        raise WorkerError("input.models must be a list")
    normalized = []
    for item in models:
        if not isinstance(item, dict):
            raise WorkerError("Each model entry must be an object")
        normalized.append(item)
    return normalized


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()

    try:
        payload = event.get("input", {}) if isinstance(event, dict) else {}

        workflow = _validate_workflow(payload.get("workflow"))
        models = _validate_models(payload.get("models"))

        downloaded: List[Dict[str, Any]] = []
        if models:
            workers = max(1, min(DOWNLOAD_WORKERS, len(models)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_download_model, model) for model in models]
                for future in as_completed(futures):
                    downloaded.append(future.result())

        _start_comfyui()
        before_output_mtime = _latest_output_mtime()

        prompt_id = _submit_prompt(workflow)
        history = _wait_for_prompt(
            prompt_id=prompt_id,
            timeout_seconds=int(payload.get("prompt_timeout_seconds", PROMPT_TIMEOUT_SECONDS)),
        )

        # Allow output writes and Telegram Sender side effects to finish before responding.
        _wait_for_output_settle(
            previous_mtime=before_output_mtime,
            timeout_seconds=int(payload.get("output_settle_timeout_seconds", 120)),
            settle_seconds=int(payload.get("output_settle_seconds", 4)),
        )

        return {
            "status": "success",
            "prompt_id": prompt_id,
            "downloaded_models": downloaded,
            "history_summary_keys": list(history.keys()),
            "duration_seconds": round(time.time() - started, 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
            "duration_seconds": round(time.time() - started, 2),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
