import glob
import os
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
COMFYUI_CPU_MODE = os.getenv("COMFYUI_CPU_MODE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
COMFYUI_HIGH_VRAM = os.getenv("COMFYUI_HIGH_VRAM", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
COMFYUI_DISABLE_SMART_MEMORY = os.getenv("COMFYUI_DISABLE_SMART_MEMORY", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
COMFYUI_FP8_E4M3FN_UNET = os.getenv("COMFYUI_FP8_E4M3FN_UNET", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}

COMFYUI_START_TIMEOUT_SECONDS = int(os.getenv("COMFYUI_START_TIMEOUT_SECONDS", "240"))
PROMPT_TIMEOUT_SECONDS = int(os.getenv("PROMPT_TIMEOUT_SECONDS", "7200"))
COMFYUI_HTTP_TIMEOUT_SECONDS = int(os.getenv("COMFYUI_HTTP_TIMEOUT_SECONDS", "30"))
TELEGRAM_HTTP_TIMEOUT_SECONDS = int(os.getenv("TELEGRAM_HTTP_TIMEOUT_SECONDS", "120"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "7200"))
DOWNLOAD_WORKERS = int(os.getenv("DOWNLOAD_WORKERS", "4"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.5"))

TELEGRAM_NODE_ID = os.getenv("TELEGRAM_NODE_ID", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_UPLOAD_ENABLED = os.getenv("TELEGRAM_UPLOAD_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
TELEGRAM_INJECT_REQUIRED = os.getenv("TELEGRAM_INJECT_REQUIRED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
OUTPUT_SETTLE_TIMEOUT_SECONDS = int(os.getenv("OUTPUT_SETTLE_TIMEOUT_SECONDS", "120"))
OUTPUT_SETTLE_SECONDS = int(os.getenv("OUTPUT_SETTLE_SECONDS", "4"))
STRICT_PROGRESS_100 = os.getenv("STRICT_PROGRESS_100", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
DELETE_UPLOADED_MP4 = os.getenv("DELETE_UPLOADED_MP4", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
OUTPUT_RETENTION_MINUTES = int(os.getenv("OUTPUT_RETENTION_MINUTES", "60"))
SERIALIZE_JOBS = os.getenv("SERIALIZE_JOBS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
WAN_I2V_MAX_BATCH_SIZE = max(1, int(os.getenv("WAN_I2V_MAX_BATCH_SIZE", "1")))
WAN_I2V_MAX_WIDTH = max(256, int(os.getenv("WAN_I2V_MAX_WIDTH", "832")))
WAN_I2V_MAX_HEIGHT = max(256, int(os.getenv("WAN_I2V_MAX_HEIGHT", "480")))
WAN_I2V_MAX_LENGTH = max(9, int(os.getenv("WAN_I2V_MAX_LENGTH", "33")))

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

# Suggestion map used for report-only node checks (no runtime install).
NODE_REPO_ALLOWLIST = {
    "VHS_VideoCombine": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    "TelegramSender": "https://github.com/matan1905/ComfyUI-Serving-Toolkit.git",
    "easy loadImageBase64": "https://github.com/yolain/ComfyUI-Easy-Use.git",
    "EmptyLTXVLatentVideo": "https://github.com/kijai/ComfyUI-KJNodes.git",
    "UnetLoaderGGUF": "https://github.com/city96/ComfyUI-GGUF.git",
    "DualCLIPLoaderGGUF": "https://github.com/city96/ComfyUI-GGUF.git",
    "CLIPLoaderGGUF": "https://github.com/city96/ComfyUI-GGUF.git",
}

_server_lock = threading.Lock()
_server_process: subprocess.Popen | None = None
_server_log_handle = None
_job_lock = threading.Lock()


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
        if COMFYUI_HIGH_VRAM:
            cmd.append("--highvram")
        if COMFYUI_DISABLE_SMART_MEMORY:
            cmd.append("--disable-smart-memory")
        if COMFYUI_FP8_E4M3FN_UNET and not COMFYUI_CPU_MODE:
            cmd.append("--fp8_e4m3fn-unet")
        if COMFYUI_CPU_MODE:
            cmd.append("--cpu")
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


def _default_clip_ref(workflow: dict[str, Any]) -> list[Any] | None:
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", "")).strip()
        if class_type in {"CLIPLoader", "CLIPLoaderGGUF", "DualCLIPLoaderGGUF"}:
            return [str(node_id), 0]
    return None


def _sanitize_workflow_inputs(workflow: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    fallback_clip = _default_clip_ref(workflow)

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", "")).strip()
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        if class_type == "easy loadImageBase64":
            base64_data = inputs.get("base64_data")
            if isinstance(base64_data, str) and base64_data.startswith("data:") and ";base64," in base64_data:
                inputs["base64_data"] = base64_data.split(";base64,", 1)[1]
                notes.append(f"node {node_id}: normalized easy loadImageBase64 base64_data")

        if class_type == "LoraLoader" and "clip" not in inputs:
            model_ref = inputs.get("model")
            fixed = False
            if isinstance(model_ref, list) and len(model_ref) >= 2:
                source_id = str(model_ref[0])
                source_node = workflow.get(source_id)
                if isinstance(source_node, dict):
                    source_class = str(source_node.get("class_type", "")).strip()
                    if source_class == "LoraLoader":
                        inputs["clip"] = [source_id, 1]
                        fixed = True
                    elif source_class in {"CLIPLoader", "CLIPLoaderGGUF", "DualCLIPLoaderGGUF"}:
                        inputs["clip"] = [source_id, 0]
                        fixed = True
            if not fixed and fallback_clip is not None:
                inputs["clip"] = [fallback_clip[0], fallback_clip[1]]
                fixed = True
            if fixed:
                notes.append(f"node {node_id}: auto-filled missing LoraLoader clip input")

        if class_type == "KSamplerAdvanced":
            add_noise = inputs.get("add_noise")
            if add_noise not in {"enable", "disable"}:
                normalized_add_noise = "enable"
                if isinstance(add_noise, bool):
                    normalized_add_noise = "enable" if add_noise else "disable"
                elif isinstance(add_noise, (int, float)):
                    normalized_add_noise = "enable" if add_noise else "disable"
                elif isinstance(add_noise, str):
                    text = add_noise.strip().lower()
                    if text in {"on", "yes", "true", "1", "enable", "enabled"}:
                        normalized_add_noise = "enable"
                    elif text in {"off", "no", "false", "0", "disable", "disabled"}:
                        normalized_add_noise = "disable"
                inputs["add_noise"] = normalized_add_noise
                notes.append(f"node {node_id}: normalized KSamplerAdvanced add_noise")

            steps = inputs.get("steps")
            if not isinstance(steps, int):
                steps = 20
                inputs["steps"] = steps
                notes.append(f"node {node_id}: defaulted KSamplerAdvanced steps to 20")

            for key, default_value in (("noise_seed", 42), ("start_at_step", 0), ("end_at_step", steps)):
                if not isinstance(inputs.get(key), int):
                    inputs[key] = default_value
                    notes.append(f"node {node_id}: defaulted KSamplerAdvanced {key}")

            cfg = inputs.get("cfg")
            if not isinstance(cfg, (int, float)):
                inputs["cfg"] = 1.0
                notes.append(f"node {node_id}: defaulted KSamplerAdvanced cfg")

            start = int(inputs.get("start_at_step", 0))
            end = int(inputs.get("end_at_step", steps))
            if end < start:
                inputs["end_at_step"] = start
                notes.append(f"node {node_id}: clamped end_at_step to start_at_step")

        if class_type == "WanImageToVideo":
            batch_size = inputs.get("batch_size")
            if not isinstance(batch_size, int) or batch_size < 1:
                inputs["batch_size"] = 1
                notes.append(f"node {node_id}: defaulted WanImageToVideo batch_size to 1")
            elif batch_size > WAN_I2V_MAX_BATCH_SIZE:
                inputs["batch_size"] = WAN_I2V_MAX_BATCH_SIZE
                notes.append(
                    f"node {node_id}: clamped WanImageToVideo batch_size to {WAN_I2V_MAX_BATCH_SIZE}"
                )

            width = inputs.get("width")
            if not isinstance(width, int) or width < 256:
                inputs["width"] = 832
                notes.append(f"node {node_id}: defaulted WanImageToVideo width to 832")
            elif width > WAN_I2V_MAX_WIDTH:
                inputs["width"] = WAN_I2V_MAX_WIDTH
                notes.append(
                    f"node {node_id}: clamped WanImageToVideo width to {WAN_I2V_MAX_WIDTH}"
                )

            height = inputs.get("height")
            if not isinstance(height, int) or height < 256:
                inputs["height"] = 480
                notes.append(f"node {node_id}: defaulted WanImageToVideo height to 480")
            elif height > WAN_I2V_MAX_HEIGHT:
                inputs["height"] = WAN_I2V_MAX_HEIGHT
                notes.append(
                    f"node {node_id}: clamped WanImageToVideo height to {WAN_I2V_MAX_HEIGHT}"
                )

            length = inputs.get("length")
            if not isinstance(length, int) or length < 9:
                inputs["length"] = 33
                notes.append(f"node {node_id}: defaulted WanImageToVideo length to 33")
            elif length > WAN_I2V_MAX_LENGTH:
                inputs["length"] = WAN_I2V_MAX_LENGTH
                notes.append(
                    f"node {node_id}: clamped WanImageToVideo length to {WAN_I2V_MAX_LENGTH}"
                )

    return notes


def _runtime_hint(error_text: str, log_tail: str) -> str | None:
    haystack = f"{error_text}\n{log_tail}".lower()
    if "no kernel image is available for execution on the device" in haystack:
        return (
            "CUDA kernel mismatch detected. This usually means the endpoint GPU architecture "
            "is newer than the CUDA/Torch build in the image. Redeploy with a compatible GPU "
            "(A100/H100/L40S) or rebuild the image with a newer Torch/CUDA stack."
        )
    if "outofmemoryerror" in haystack or "out of memory" in haystack:
        return (
            "GPU out-of-memory detected. Reduce workflow memory pressure by lowering "
            "WanImageToVideo batch_size (recommended 1), frame length, or resolution. "
            "Current worker clamps "
            f"WAN_I2V_MAX_BATCH_SIZE={WAN_I2V_MAX_BATCH_SIZE}, "
            f"WAN_I2V_MAX_WIDTH={WAN_I2V_MAX_WIDTH}, "
            f"WAN_I2V_MAX_HEIGHT={WAN_I2V_MAX_HEIGHT}, "
            f"WAN_I2V_MAX_LENGTH={WAN_I2V_MAX_LENGTH}."
        )
    return None


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


def _workflow_class_types(workflow: dict[str, Any]) -> set[str]:
    class_types: set[str] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if isinstance(class_type, str) and class_type.strip():
            class_types.add(class_type.strip())
    return class_types


def _installed_class_types() -> set[str]:
    response = _request_with_retry("GET", f"{COMFYUI_API}/object_info", max_attempts=3, timeout=20)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise WorkerError("ComfyUI /object_info did not return a JSON object")
    return set(data.keys())


def _check_workflow_nodes(workflow: dict[str, Any]) -> dict[str, Any]:
    required_types = _workflow_class_types(workflow)
    if not required_types:
        return {
            "required_node_types": 0,
            "suggested_repos": [],
            "missing_before": [],
            "unmapped_nodes": [],
        }

    installed_before = _installed_class_types()
    missing_before = sorted(required_types - installed_before)
    suggested_repos = sorted(
        {
            NODE_REPO_ALLOWLIST[node_type]
            for node_type in missing_before
            if node_type in NODE_REPO_ALLOWLIST
        }
    )
    unresolved = sorted(node_type for node_type in missing_before if node_type not in NODE_REPO_ALLOWLIST)

    return {
        "required_node_types": len(required_types),
        "suggested_repos": suggested_repos,
        "missing_before": missing_before,
        "unmapped_nodes": unresolved,
    }


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


def _wait_for_prompt_completion(prompt_id: str, timeout_seconds: int) -> tuple[dict[str, Any], bool]:
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

                if not seen_progress_100 and STRICT_PROGRESS_100:
                    raise WorkerError(
                        "Render completed but progress never reported 100%. "
                        "Aborting upload to satisfy 100% completion requirement."
                    )
                return item, seen_progress_100

        time.sleep(1)

    raise WorkerError(f"Prompt timeout reached for {prompt_id}")


def _newest_mp4_file() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = glob.glob(str(OUTPUT_DIR / "**" / "*.mp4"), recursive=True)
    if not candidates:
        raise WorkerError(f"No .mp4 files found in {OUTPUT_DIR}")
    latest = max(candidates, key=lambda p: Path(p).stat().st_mtime)
    return Path(latest)


def _snapshot_mp4_state() -> dict[str, tuple[float, int]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, tuple[float, int]] = {}
    for path in glob.glob(str(OUTPUT_DIR / "**" / "*.mp4"), recursive=True):
        try:
            p = Path(path)
            stat = p.stat()
            state[str(p)] = (stat.st_mtime, stat.st_size)
        except OSError:
            continue
    return state


def _select_generated_mp4(before_state: dict[str, tuple[float, int]], earliest_mtime: float) -> Path:
    candidates: list[tuple[Path, float]] = []
    for path in glob.glob(str(OUTPUT_DIR / "**" / "*.mp4"), recursive=True):
        p = Path(path)
        try:
            stat = p.stat()
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError:
            continue

        previous_state = before_state.get(str(p))
        is_new = previous_state is None
        is_updated = previous_state is not None and (mtime, size) != previous_state
        is_recent = mtime >= earliest_mtime

        if (is_new or is_updated) and is_recent:
            candidates.append((p, mtime))

    if not candidates:
        raise WorkerError(
            "No new/updated MP4 detected for this request. "
            "If endpoint concurrency >1, set concurrency to 1 or SERIALIZE_JOBS=true."
        )

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def _prune_old_outputs(retention_minutes: int) -> int:
    if retention_minutes <= 0:
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - (retention_minutes * 60)
    deleted = 0
    for path in glob.glob(str(OUTPUT_DIR / "**" / "*.mp4"), recursive=True):
        p = Path(path)
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                deleted += 1
        except OSError:
            continue
    return deleted


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
    if SERIALIZE_JOBS:
        with _job_lock:
            return _handle_impl(event)
    return _handle_impl(event)


def _handle_impl(event: dict[str, Any]) -> dict[str, Any]:
    started = time.time()

    try:
        payload = event.get("input", {}) if isinstance(event, dict) else {}
        workflow = _validate_workflow(payload.get("workflow"))
        workflow_sanitization_notes = _sanitize_workflow_inputs(workflow)
        models = _validate_models(payload.get("models"))
        _prune_old_outputs(OUTPUT_RETENTION_MINUTES)

        _start_comfyui()
        node_check_report = _check_workflow_nodes(workflow)
        if node_check_report["missing_before"]:
            return {
                "status": "error",
                "error": (
                    "Workflow contains missing ComfyUI node class types. "
                    "Pre-bake suggested repos in Dockerfile and rebuild image."
                ),
                "node_check_report": node_check_report,
                "duration_seconds": round(time.time() - started, 2),
            }

        downloaded_models: list[dict[str, Any]] = []
        if models:
            workers = max(1, min(DOWNLOAD_WORKERS, len(models)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_download_model, model) for model in models]
                for future in as_completed(futures):
                    downloaded_models.append(future.result())

        _inject_telegram_if_present(workflow)
        before_output_state = _snapshot_mp4_state()
        submit_started = time.time()
        prompt_id = _submit_prompt(workflow)
        history, progress_100_observed = _wait_for_prompt_completion(
            prompt_id=prompt_id,
            timeout_seconds=int(payload.get("prompt_timeout_seconds", PROMPT_TIMEOUT_SECONDS)),
        )

        output_selection_fallback = False
        try:
            newest_mp4 = _select_generated_mp4(
                before_state=before_output_state,
                earliest_mtime=submit_started,
            )
        except WorkerError:
            # Local bind mounts can have timestamp behavior that makes strict
            # "new since submit" detection unreliable. Fall back to newest MP4.
            newest_mp4 = _newest_mp4_file()
            output_selection_fallback = True
        _wait_for_mp4_settle(
            file_path=newest_mp4,
            timeout_seconds=OUTPUT_SETTLE_TIMEOUT_SECONDS,
            settle_seconds=OUTPUT_SETTLE_SECONDS,
        )
        telegram_result: dict[str, Any] | None = None
        telegram_skipped = False
        if TELEGRAM_UPLOAD_ENABLED:
            telegram_result = _send_video_to_telegram(
                video_path=newest_mp4,
                caption=str(payload.get("telegram_caption", "RunPod render completed")),
            )
        else:
            telegram_skipped = True

        if DELETE_UPLOADED_MP4:
            try:
                newest_mp4.unlink(missing_ok=True)
            except OSError:
                pass

        return {
            "status": "success",
            "prompt_id": prompt_id,
            "progress_100_observed": progress_100_observed,
            "strict_progress_100": STRICT_PROGRESS_100,
            "workflow_sanitization_notes": workflow_sanitization_notes,
            "node_check_report": node_check_report,
            "output_selection_fallback": output_selection_fallback,
            "mp4_path": str(newest_mp4),
            "downloaded_models": downloaded_models,
            "history_summary_keys": list(history.keys()) if isinstance(history, dict) else [],
            "telegram_upload_enabled": TELEGRAM_UPLOAD_ENABLED,
            "telegram_skipped": telegram_skipped,
            "telegram_message_id": (
                telegram_result.get("result", {}).get("message_id")
                if isinstance(telegram_result, dict)
                else None
            ),
            "duration_seconds": round(time.time() - started, 2),
        }

    except Exception as exc:  # noqa: BLE001
        comfyui_tail = _log_tail()
        return {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
            "comfyui_log_tail": comfyui_tail,
            "runtime_hint": _runtime_hint(str(exc), comfyui_tail),
            "duration_seconds": round(time.time() - started, 2),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
