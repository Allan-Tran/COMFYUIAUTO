import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gradio as gr
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=False)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"
WORKFLOW_PATH = Path("workflow_api.json")
MODELS_DEFAULTS_PATH = Path("models_defaults.json")
MODELS_JSON_PATH = Path(os.getenv("MODELS_JSON_PATH", "input_models.json"))
HF_API = HfApi()
RUNPOD_STATUS_POLL_TIMEOUT_SECONDS = int(os.getenv("RUNPOD_STATUS_POLL_TIMEOUT_SECONDS", "45"))
RUNPOD_STATUS_POLL_INTERVAL_SECONDS = float(os.getenv("RUNPOD_STATUS_POLL_INTERVAL_SECONDS", "2"))
MODEL_FILE_EXTENSIONS = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin")
MODEL_HINT_KEYS = {
    "ckpt_name",
    "vae_name",
    "unet_name",
    "clip_name",
    "clip_l_name",
    "clip_g_name",
    "t5_name",
    "text_encoder",
    "text_encoder_name",
    "model_name",
    "lora_name",
    "control_net_name",
    "controlnet_name",
    "diffusion_model",
}
MODEL_TYPE_CHOICES = [
    "diffusion_models",
    "checkpoints",
    "text_encoders",
    "unet",
    "vae",
    "clip",
    "clip_vision",
    "loras",
    "controlnet",
    "upscale_models",
]

# Optional node mapping via .env for direct assignment.
POSITIVE_NODE_ID = os.getenv("POSITIVE_NODE_ID", "")
POSITIVE_INPUT_KEY = os.getenv("POSITIVE_INPUT_KEY", "text")
NEGATIVE_NODE_ID = os.getenv("NEGATIVE_NODE_ID", "")
NEGATIVE_INPUT_KEY = os.getenv("NEGATIVE_INPUT_KEY", "text")
LENGTH_NODE_ID = os.getenv("LENGTH_NODE_ID", "")
LENGTH_INPUT_KEY = os.getenv("LENGTH_INPUT_KEY", "length")
I2V_NODE_ID = os.getenv("I2V_NODE_ID", "")
I2V_INPUT_KEY = os.getenv("I2V_INPUT_KEY", "image_base64")
TELEGRAM_NODE_ID = os.getenv("TELEGRAM_NODE_ID", "")
TELEGRAM_INJECT_REQUIRED = os.getenv("TELEGRAM_INJECT_REQUIRED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}

# Placeholder strategy works even if node IDs change.
PLACEHOLDERS = {
    "__POSITIVE_PROMPT__": lambda ctx: ctx["positive_prompt"],
    "__NEGATIVE_PROMPT__": lambda ctx: ctx["negative_prompt"],
    "__FRAME_LENGTH__": lambda ctx: str(ctx["frame_length"]),
    "__TELEGRAM_BOT_TOKEN__": lambda ctx: ctx["telegram_bot_token"],
    "__TELEGRAM_CHAT_ID__": lambda ctx: ctx["telegram_chat_id"],
    "__I2V_IMAGE_B64__": lambda ctx: ctx["image_b64"],
}


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_default_models() -> list[dict[str, str]]:
    if not MODELS_DEFAULTS_PATH.exists():
        return []

    try:
        raw = json.loads(MODELS_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {MODELS_DEFAULTS_PATH}: {exc}") from exc

    if not isinstance(raw, list):
        raise RuntimeError("models_defaults.json must be a list")

    models: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeError(f"models_defaults.json[{idx}] must be an object")
        model_type = str(item.get("type", "")).strip()
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        if not model_type or not name or not url:
            raise RuntimeError(f"models_defaults.json[{idx}] requires type, name, and url")
        models.append({"type": model_type, "name": name, "url": url})

    return models


def load_models_from_json_file(models_path: Path) -> list[dict[str, str]]:
    if not models_path.exists():
        return []

    try:
        raw = json.loads(models_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {models_path}: {exc}") from exc

    if not isinstance(raw, list):
        raise RuntimeError(f"{models_path} must be a JSON list")

    models: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeError(f"{models_path}[{idx}] must be an object")
        model_type = str(item.get("type", "")).strip()
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        if not model_type or not name or not url:
            raise RuntimeError(f"{models_path}[{idx}] requires type, name, and url")
        if "replace-me" in url or "example.com" in url:
            raise RuntimeError(f"{models_path}[{idx}] has placeholder url for {name}: {url}")
        models.append({"type": model_type, "name": name, "url": url})

    return models


def get_hf_token() -> str:
    # Supports both names for convenience. HUGGING_FACE_HUB_TOKEN takes priority.
    return os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip() or os.getenv("HF_TOKEN", "").strip()


def build_hf_resolve_url(repo_id: str, file_path: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{quote(file_path, safe='/')}"


def _parse_workflow_from_input(uploaded_file_path: str | None, pasted_json: str | None) -> dict[str, Any]:
    content = ""
    if uploaded_file_path:
        path = Path(uploaded_file_path)
        if not path.exists():
            raise RuntimeError(f"Uploaded workflow file not found: {uploaded_file_path}")
        content = path.read_text(encoding="utf-8")
    elif (pasted_json or "").strip():
        content = pasted_json.strip()
    else:
        if not WORKFLOW_PATH.exists():
            raise RuntimeError(
                "No workflow provided. Upload/paste a workflow JSON or add local workflow_api.json."
            )
        content = WORKFLOW_PATH.read_text(encoding="utf-8")

    raw = json.loads(content)
    if isinstance(raw, dict) and isinstance(raw.get("input"), dict) and isinstance(raw["input"].get("workflow"), dict):
        return raw["input"]["workflow"]
    if isinstance(raw, dict) and raw:
        return raw
    raise RuntimeError("Workflow JSON must be a JSON object or include input.workflow")


def _is_model_candidate(key: str, value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    lower_v = v.lower()
    lower_key = (key or "").strip().lower()
    if lower_v.endswith(MODEL_FILE_EXTENSIONS):
        return True
    if lower_key in MODEL_HINT_KEYS:
        # For hint keys, require filename-like content to avoid false positives like "cpu".
        file_name = Path(v).name
        return "." in file_name
    return False


def _normalize_model_query(value: str) -> str:
    val = (value or "").strip()
    if not val:
        return ""
    if "/" in val and " " not in val and not val.lower().endswith(MODEL_FILE_EXTENSIONS):
        return val
    return Path(val).name


def _collect_model_candidates(value: Any, path: str, out: list[dict[str, str]], key_hint: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _collect_model_candidates(child, child_path, out, key_hint=str(key))
        return

    if isinstance(value, list):
        for idx, child in enumerate(value):
            child_path = f"{path}[{idx}]"
            _collect_model_candidates(child, child_path, out, key_hint=key_hint)
        return

    if isinstance(value, str) and _is_model_candidate(key_hint, value):
        query = _normalize_model_query(value)
        if query:
            out.append({"query": query, "value": value, "path": path, "key": key_hint})


def _guess_model_type_from_key(key_hint: str) -> str:
    key_lower = (key_hint or "").lower()
    if "vae" in key_lower:
        return "vae"
    if "lora" in key_lower:
        return "loras"
    if "control" in key_lower:
        return "controlnet"
    if "clip_vision" in key_lower:
        return "clip_vision"
    if "clip" in key_lower:
        return "clip"
    if "text_encoder" in key_lower or "t5" in key_lower:
        return "text_encoders"
    if "unet" in key_lower:
        return "unet"
    if "upscale" in key_lower:
        return "upscale_models"
    return "diffusion_models"


def _resolve_hf_url_for_filename(file_name: str) -> str | None:
    token = get_hf_token() or None
    try:
        repos = [m.id for m in HF_API.list_models(search=file_name, limit=10)]
    except Exception:
        return None

    lowered = file_name.lower()
    for repo_id in repos:
        try:
            files = HF_API.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
        except Exception:
            continue
        for path in files:
            if Path(path).name.lower() == lowered:
                return build_hf_resolve_url(repo_id, path)
    return None


def auto_build_models_from_workflow(
    uploaded_file_path: str | None,
    pasted_json: str,
) -> tuple[list[dict[str, str]], str, str]:
    try:
        workflow = _parse_workflow_from_input(uploaded_file_path, pasted_json)
    except Exception as exc:  # noqa: BLE001
        return [], "[]", f"Red Alert: {exc}"

    raw_hits: list[dict[str, str]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        _collect_model_candidates(inputs, path=f"node[{node_id}].inputs", out=raw_hits)

    # Deduplicate by (key, filename) to reduce duplicate URL lookups.
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for hit in raw_hits:
        name = Path(hit["value"]).name
        if not name:
            continue
        unique[(hit.get("key", ""), name)] = hit

    if not unique:
        return [], "[]", "No model-like references found to auto-build input.models."

    included_names = set()
    try:
        included_names = {
            Path(str(model.get("name", ""))).name.lower()
            for model in load_models_from_json_file(MODELS_JSON_PATH)
            if str(model.get("name", "")).strip()
        }
    except Exception:
        included_names = set()

    models: list[dict[str, str]] = []
    unresolved: list[str] = []
    for hit in sorted(unique.values(), key=lambda h: Path(h["value"]).name.lower()):
        model_name = Path(hit["value"]).name
        url = _resolve_hf_url_for_filename(model_name)
        if not url:
            if model_name.lower() not in included_names:
                unresolved.append(model_name)
            continue
        models.append(
            {
                "type": _guess_model_type_from_key(hit.get("key", "")),
                "name": model_name,
                "url": url,
            }
        )

    preview = json.dumps(models, indent=2)
    status = f"Auto-built {len(models)} model entries from workflow references."
    if unresolved:
        status += f" Unresolved filenames: {sorted(set(unresolved))}"
    return models, preview, status


def scan_workflow_for_models(
    uploaded_file_path: str | None,
    pasted_json: str,
) -> tuple[gr.Dropdown, dict[str, dict[str, str]], str]:
    try:
        workflow = _parse_workflow_from_input(uploaded_file_path, pasted_json)
    except Exception as exc:  # noqa: BLE001
        return gr.update(choices=[], value=None), {}, f"Red Alert: {exc}"

    raw_hits: list[dict[str, str]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        _collect_model_candidates(inputs, path=f"node[{node_id}].inputs", out=raw_hits)

    deduped: dict[tuple[str, str], dict[str, str]] = {}
    for hit in raw_hits:
        key = (hit["query"], hit["path"])
        deduped[key] = hit

    if not deduped:
        return (
            gr.update(choices=[], value=None),
            {},
            "No model-like references found in workflow inputs (.safetensors/.gguf or known model keys).",
        )

    items = sorted(deduped.values(), key=lambda item: (item["query"].lower(), item["path"].lower()))
    state: dict[str, dict[str, str]] = {}
    choices: list[str] = []
    for item in items:
        label = f"{item['query']}  [{item['path']}]"
        state[label] = item
        choices.append(label)

    note = (
        f"Discovered {len(choices)} model reference(s). Select one and click Search Selected Workflow Model."
    )
    return gr.update(choices=choices, value=choices[0]), state, note


def search_selected_workflow_model(
    selected_workflow_model: str,
    workflow_scan_state: dict[str, dict[str, str]],
) -> tuple[str, gr.Dropdown, dict[str, dict[str, Any]], str]:
    if not selected_workflow_model:
        return "", gr.update(choices=[], value=None), {}, "Pick a discovered workflow model first."

    if (
        not isinstance(workflow_scan_state, dict)
        or selected_workflow_model not in workflow_scan_state
    ):
        return "", gr.update(choices=[], value=None), {}, "Selection missing. Run Scan Workflow Models again."

    query = workflow_scan_state[selected_workflow_model].get("query", "").strip()
    if not query:
        return "", gr.update(choices=[], value=None), {}, "Selected model query is empty."

    dropdown_update, search_state, status = search_hf_models(query)
    return query, dropdown_update, search_state, f"Workflow model search for '{query}': {status}"


def _collect_repo_files(repo_id: str, token: str) -> list[dict[str, Any]]:
    token_value = token or None
    info = HF_API.model_info(repo_id=repo_id, token=token_value)
    files = HF_API.list_repo_files(repo_id=repo_id, repo_type="model", token=token_value)

    result = []
    for file_name in files:
        if file_name.lower().endswith((".safetensors", ".gguf")):
            gated = bool(getattr(info, "gated", False))
            gate_tag = " [GATED]" if gated else ""
            label = f"{repo_id} :: {file_name}{gate_tag}"
            result.append(
                {
                    "label": label,
                    "repo_id": repo_id,
                    "file_name": file_name,
                    "url": build_hf_resolve_url(repo_id, file_name),
                    "gated": gated,
                }
            )
    return result


def search_hf_models(search_text: str) -> tuple[gr.Dropdown, dict[str, dict[str, Any]], str]:
    query = (search_text or "").strip()
    if not query:
        return gr.update(choices=[], value=None), {}, "Enter a model name or repo ID first."

    token = get_hf_token()
    repos: list[str]
    if "/" in query and " " not in query:
        repos = [query]
    else:
        repos = [m.id for m in HF_API.list_models(search=query, limit=8)]

    found: list[dict[str, Any]] = []
    errors: list[str] = []

    for repo_id in repos:
        try:
            found.extend(_collect_repo_files(repo_id, token=token))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{repo_id}: {exc}")

    if not found:
        error_block = "\n".join(f"- {e}" for e in errors[:5]) if errors else "- No matching files found"
        return (
            gr.update(choices=[], value=None),
            {},
            "Red Alert: no .safetensors/.gguf files found or repos are inaccessible.\n" + error_block,
        )

    found.sort(key=lambda item: item["label"].lower())
    state = {item["label"]: item for item in found}
    choices = list(state.keys())
    status = f"Found {len(choices)} candidate files across {len(repos)} repo(s)."
    gated_count = sum(1 for item in found if item.get("gated"))
    if gated_count:
        if token:
            status += f" {gated_count} result(s) are gated. Token detected for access attempts."
        else:
            status += (
                f" {gated_count} result(s) are gated. Add HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) to .env."
            )
    if errors:
        status += f" Some repos failed to list ({len(errors)})."
    return gr.update(choices=choices, value=choices[0]), state, status


def validate_selected_model(
    selected_label: str,
    search_state: dict[str, dict[str, Any]],
    selected_model_type: str,
    min_size_gb: float,
) -> tuple[bool, str, dict[str, str] | None]:
    if not selected_label:
        return True, "No custom model selected.", None

    if not isinstance(search_state, dict) or selected_label not in search_state:
        return False, "Red Alert: selected model is missing. Click Search Models again.", None

    model_meta = search_state[selected_label]
    token = get_hf_token()

    if model_meta.get("gated") and not token:
        return (
            False,
            "Red Alert: selected model is gated and requires HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) in .env.",
            None,
        )

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.head(model_meta["url"], headers=headers, allow_redirects=True, timeout=45)
    except requests.RequestException as exc:
        return False, f"Red Alert: model HEAD check failed: {exc}", None

    if response.status_code in {401, 403}:
        return (
            False,
            "Red Alert: access denied for this model URL. Check token permissions or gated access approval.",
            None,
        )

    if response.status_code != 200:
        return (
            False,
            f"Red Alert: model URL returned HTTP {response.status_code}. Submission blocked.",
            None,
        )

    content_length = response.headers.get("Content-Length", "").strip()
    min_bytes = int(float(min_size_gb) * (1024**3))
    if content_length.isdigit():
        size_bytes = int(content_length)
        if size_bytes < min_bytes:
            return (
                False,
                f"Red Alert: model is too small ({size_bytes} bytes). Expected at least {min_bytes} bytes.",
                None,
            )
        size_note = f"{size_bytes} bytes"
    else:
        size_note = "size unknown (no Content-Length header)"

    model_payload = {
        "type": selected_model_type,
        "name": Path(model_meta["file_name"]).name,
        "url": model_meta["url"],
    }
    return True, f"Pre-flight OK: {model_payload['name']} ({size_note}).", model_payload


def upsert_model(models: list[dict[str, str]], new_model: dict[str, str]) -> list[dict[str, str]]:
    updated = [dict(item) for item in models]
    for idx, model in enumerate(updated):
        if model.get("type") == new_model.get("type"):
            updated[idx] = new_model
            return updated
    updated.append(new_model)
    return updated


def append_unique_models(models: list[dict[str, str]], additions: list[dict[str, str]]) -> list[dict[str, str]]:
    updated = [dict(item) for item in models]
    seen = {(str(m.get("type")), str(m.get("name")), str(m.get("url"))) for m in updated}
    for item in additions:
        key = (str(item.get("type")), str(item.get("name")), str(item.get("url")))
        if key in seen:
            continue
        updated.append({"type": key[0], "name": key[1], "url": key[2]})
        seen.add(key)
    return updated


def load_workflow() -> dict[str, Any]:
    if not WORKFLOW_PATH.exists():
        raise RuntimeError(f"workflow_api.json not found at {WORKFLOW_PATH}")

    raw = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and isinstance(raw.get("input"), dict) and isinstance(raw["input"].get("workflow"), dict):
        return raw["input"]["workflow"]

    if isinstance(raw, dict) and raw:
        return raw

    raise RuntimeError("workflow_api.json must be a JSON object or include input.workflow")


def _find_node_id_by_class_title(workflow: dict[str, Any], class_type: str, title_keyword: str) -> str | None:
    title_matches: list[str] = []
    class_matches: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")).strip() != class_type:
            continue
        class_matches.append(node_id)
        meta = node.get("_meta")
        title = ""
        if isinstance(meta, dict):
            title = str(meta.get("title", "")).lower()
        if title_keyword.lower() in title:
            title_matches.append(node_id)

    if len(title_matches) == 1:
        return title_matches[0]
    if len(class_matches) == 1:
        return class_matches[0]
    return None


def _detect_length_node_id(workflow: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if "length" in inputs:
            candidates.append(node_id)
    if len(candidates) == 1:
        return candidates[0]
    return None


def resolve_prompt_node_ids(workflow: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    positive_node_id = POSITIVE_NODE_ID or _find_node_id_by_class_title(
        workflow, class_type="CLIPTextEncode", title_keyword="positive"
    )
    negative_node_id = NEGATIVE_NODE_ID or _find_node_id_by_class_title(
        workflow, class_type="CLIPTextEncode", title_keyword="negative"
    )
    length_node_id = LENGTH_NODE_ID or _detect_length_node_id(workflow)
    return positive_node_id, negative_node_id, length_node_id


def resolve_telegram_node_id(workflow: dict[str, Any]) -> str | None:
    if TELEGRAM_NODE_ID:
        if TELEGRAM_NODE_ID not in workflow:
            raise RuntimeError(f"TELEGRAM_NODE_ID {TELEGRAM_NODE_ID!r} was not found in workflow")
        return TELEGRAM_NODE_ID

    sender_ids = [
        node_id
        for node_id, node in workflow.items()
        if isinstance(node, dict) and str(node.get("class_type", "")).strip() == "TelegramSender"
    ]
    if len(sender_ids) == 1:
        return sender_ids[0]
    if len(sender_ids) > 1:
        if TELEGRAM_INJECT_REQUIRED:
            raise RuntimeError("Multiple TelegramSender nodes found. Set TELEGRAM_NODE_ID in .env.")
        return None

    if TELEGRAM_INJECT_REQUIRED:
        raise RuntimeError("TelegramSender node not found. Set TELEGRAM_NODE_ID in .env.")
    return None


def set_node_input(workflow: dict[str, Any], node_id: str, input_key: str, value: Any) -> None:
    if not node_id:
        return
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        raise RuntimeError(f"Node {node_id!r} not found or invalid in workflow")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
        node["inputs"] = inputs
    inputs[input_key] = value


def replace_placeholders(obj: Any, context: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: replace_placeholders(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_placeholders(v, context) for v in obj]
    if isinstance(obj, str):
        value = obj
        for token, fn in PLACEHOLDERS.items():
            replacement = fn(context)
            value = value.replace(token, replacement)
        if obj == "__FRAME_LENGTH__":
            return int(value)
        return value
    return obj


def image_to_data_uri(image_path: str | None) -> str:
    if not image_path:
        return ""
    path = Path(image_path)
    if not path.exists():
        raise RuntimeError(f"Uploaded image path does not exist: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def data_uri_to_base64(data_uri: str) -> str:
    value = (data_uri or "").strip()
    marker = ";base64,"
    if not value:
        return ""
    if value.startswith("data:") and marker in value:
        return value.split(marker, 1)[1]
    return value


def resolve_easy_load_image_base64_node_id(workflow: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", "")).strip()
        if class_type != "easy loadImageBase64":
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "base64_data" in inputs:
            candidates.append(node_id)
    if len(candidates) == 1:
        return candidates[0]
    return None


def validate_workflow_runtime_compat(workflow: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", "")).strip()
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        if class_type == "LoraLoader":
            if "model" not in inputs:
                issues.append(f"node {node_id} (LoraLoader): missing required input 'model'")
            if "clip" not in inputs:
                issues.append(f"node {node_id} (LoraLoader): missing required input 'clip'")

        if class_type == "KSamplerAdvanced":
            add_noise = inputs.get("add_noise")
            if add_noise not in {"enable", "disable"}:
                issues.append(
                    f"node {node_id} (KSamplerAdvanced): add_noise must be 'enable' or 'disable'"
                )
            for key in ("noise_seed", "start_at_step", "end_at_step", "steps"):
                if not isinstance(inputs.get(key), int):
                    issues.append(f"node {node_id} (KSamplerAdvanced): {key} must be an integer")
            if not isinstance(inputs.get("cfg"), (int, float)):
                issues.append(f"node {node_id} (KSamplerAdvanced): cfg must be numeric")

    return issues


def submit(payload: dict[str, Any], endpoint_id: str, api_key: str) -> dict[str, Any]:
    url = f"{RUNPOD_API_BASE}/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"RunPod API error {response.status_code}: {response.text[:1200]}")

    data = response.json()
    return data


def get_job_status(endpoint_id: str, api_key: str, job_id: str) -> dict[str, Any]:
    url = f"{RUNPOD_API_BASE}/{endpoint_id}/status/{job_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"RunPod status API error {response.status_code}: {response.text[:1200]}")
    return response.json()


def format_node_check_report(report: dict[str, Any]) -> str:
    missing = report.get("missing_before") or []
    suggested = report.get("suggested_repos") or []
    unmapped = report.get("unmapped_nodes") or []

    lines = ["Missing ComfyUI Nodes Detected:"]
    lines.append(f"missing_before: {missing}")
    lines.append(f"suggested_repos: {suggested}")
    lines.append(f"unmapped_nodes: {unmapped}")
    return "\n".join(lines)


def _clip_text(text: Any, limit: int = 2400) -> str:
    if text is None:
        return ""
    value = str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...<truncated>"


def poll_for_terminal_status(endpoint_id: str, api_key: str, job_id: str) -> dict[str, Any] | None:
    deadline = time.time() + RUNPOD_STATUS_POLL_TIMEOUT_SECONDS
    terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

    while time.time() < deadline:
        status_payload = get_job_status(endpoint_id=endpoint_id, api_key=api_key, job_id=job_id)
        state = str(status_payload.get("status", "")).upper()

        if state in terminal_states:
            return status_payload

        time.sleep(RUNPOD_STATUS_POLL_INTERVAL_SECONDS)

    return None


def build_startup_status_markdown() -> str:
    warnings: list[str] = []

    for name in [
        "RUNPOD_API_KEY",
        "RUNPOD_ENDPOINT_ID",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
    ]:
        if not os.getenv(name, "").strip():
            warnings.append(f"Missing required .env value: {name}")

    workflow: dict[str, Any] | None = None
    try:
        workflow = load_workflow()
    except Exception as exc:  # noqa: BLE001
        warnings.append(str(exc))

    try:
        load_default_models()
    except Exception as exc:  # noqa: BLE001
        warnings.append(str(exc))

    try:
        load_models_from_json_file(MODELS_JSON_PATH)
    except Exception as exc:  # noqa: BLE001
        warnings.append(str(exc))

    if workflow is not None:
        runtime_issues = validate_workflow_runtime_compat(workflow)
        warnings.extend(runtime_issues)

        positive_node_id, negative_node_id, length_node_id = resolve_prompt_node_ids(workflow)
        for node_id, input_key, label in [
            (positive_node_id, POSITIVE_INPUT_KEY, "POSITIVE"),
            (negative_node_id, NEGATIVE_INPUT_KEY, "NEGATIVE"),
            (length_node_id, LENGTH_INPUT_KEY, "LENGTH"),
        ]:
            if not node_id:
                warnings.append(
                    f"Could not auto-detect {label} node. Set {label}_NODE_ID in .env for this workflow."
                )
                continue
            node = workflow.get(node_id)
            if not isinstance(node, dict):
                warnings.append(f"{label}_NODE_ID={node_id!r} not found in workflow")
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or input_key not in inputs:
                warnings.append(
                    f"{label}_INPUT_KEY={input_key!r} not present on node {node_id!r}"
                )

        if I2V_NODE_ID:
            node = workflow.get(I2V_NODE_ID)
            if not isinstance(node, dict):
                warnings.append(f"I2V_NODE_ID={I2V_NODE_ID!r} not found in workflow")
            else:
                inputs = node.get("inputs")
                if not isinstance(inputs, dict) or I2V_INPUT_KEY not in inputs:
                    warnings.append(
                        f"I2V_INPUT_KEY={I2V_INPUT_KEY!r} not present on node {I2V_NODE_ID!r}"
                    )

        try:
            telegram_node_id = resolve_telegram_node_id(workflow)
            if TELEGRAM_INJECT_REQUIRED and not telegram_node_id:
                warnings.append("TELEGRAM_INJECT_REQUIRED=true but TelegramSender node is missing")
        except Exception as exc:  # noqa: BLE001
            warnings.append(str(exc))

    if not warnings:
        return "### Startup Check: Ready\nAll required files and configuration values look good."

    lines = "\n".join(f"- {item}" for item in warnings)
    return "### Startup Check: Attention Needed\nFix the following before running a job:\n" + lines


def trigger_job(
    positive_prompt: str,
    negative_prompt: str,
    frame_length: int,
    image_path: str | None,
    workflow_file_path: str | None,
    workflow_json_text: str,
    selected_model_label: str,
    selected_model_type: str,
    min_model_size_gb: float,
    search_state: dict[str, dict[str, Any]],
    use_auto_models: bool,
    auto_models_state: list[dict[str, str]],
) -> str:
    try:
        api_key = require_env("RUNPOD_API_KEY")
        endpoint_id = require_env("RUNPOD_ENDPOINT_ID")
        telegram_bot_token = require_env("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = require_env("TELEGRAM_CHAT_ID")

        workflow = _parse_workflow_from_input(workflow_file_path, workflow_json_text)
        positive_node_id, negative_node_id, length_node_id = resolve_prompt_node_ids(workflow)

        # Direct node-id injection path.
        if not positive_node_id or not negative_node_id or not length_node_id:
            raise RuntimeError(
                "Could not resolve prompt node IDs automatically. Set POSITIVE_NODE_ID, "
                "NEGATIVE_NODE_ID, and LENGTH_NODE_ID in .env for this workflow."
            )
        set_node_input(workflow, positive_node_id, POSITIVE_INPUT_KEY, positive_prompt)
        set_node_input(workflow, negative_node_id, NEGATIVE_INPUT_KEY, negative_prompt)
        set_node_input(workflow, length_node_id, LENGTH_INPUT_KEY, int(frame_length))

        image_b64_data_uri = image_to_data_uri(image_path)
        image_b64_raw = data_uri_to_base64(image_b64_data_uri)
        if image_b64_data_uri and I2V_NODE_ID:
            value = image_b64_raw if I2V_INPUT_KEY == "base64_data" else image_b64_data_uri
            set_node_input(workflow, I2V_NODE_ID, I2V_INPUT_KEY, value)

        easy_load_image_node_id = resolve_easy_load_image_base64_node_id(workflow)
        if image_b64_raw and easy_load_image_node_id:
            set_node_input(workflow, easy_load_image_node_id, "base64_data", image_b64_raw)

        # Telegram credentials are injected at request time so they are never stored in workflow_api.json.
        telegram_node_id = resolve_telegram_node_id(workflow)
        if telegram_node_id:
            set_node_input(workflow, telegram_node_id, "bot_token", telegram_bot_token)
            set_node_input(workflow, telegram_node_id, "chat_id", telegram_chat_id)

        # Placeholder replacement fallback for workflows built with tokens.
        context = {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "frame_length": str(int(frame_length)),
            "telegram_bot_token": telegram_bot_token,
            "telegram_chat_id": telegram_chat_id,
            "image_b64": image_b64_data_uri,
        }
        workflow = replace_placeholders(workflow, context)
        runtime_issues = validate_workflow_runtime_compat(workflow)
        if runtime_issues:
            issues_block = "\n- " + "\n- ".join(runtime_issues)
            return "Red Alert: workflow validation failed before submit." + issues_block

        models = load_default_models()

        file_models = load_models_from_json_file(MODELS_JSON_PATH)
        if file_models:
            models = append_unique_models(models, file_models)

        if use_auto_models and isinstance(auto_models_state, list):
            models = append_unique_models(models, auto_models_state)

        is_valid, validation_message, custom_model = validate_selected_model(
            selected_label=selected_model_label,
            search_state=search_state,
            selected_model_type=selected_model_type,
            min_size_gb=min_model_size_gb,
        )
        if not is_valid:
            return validation_message

        if custom_model is not None:
            models = upsert_model(models, custom_model)

        payload = {
            "input": {
                "workflow": workflow,
                "models": models,
            }
        }

        result = submit(payload=payload, endpoint_id=endpoint_id, api_key=api_key)
        job_id = result.get("id") or result.get("jobId") or result.get("requestId")
        if not job_id:
            return f"Request sent, but no job id returned. Response: {result}"

        terminal_status = None
        try:
            terminal_status = poll_for_terminal_status(
                endpoint_id=endpoint_id,
                api_key=api_key,
                job_id=str(job_id),
            )
        except Exception as poll_exc:  # noqa: BLE001
            terminal_status = {
                "status": "UNKNOWN",
                "output": {
                    "error": f"Status polling warning: {poll_exc}",
                },
            }

        dashboard = f"https://www.runpod.io/console/serverless/user/endpoints/{endpoint_id}"
        status_api = f"{RUNPOD_API_BASE}/{endpoint_id}/status/{job_id}"
        message = (
            f"Submitted successfully.\n"
            f"Job ID: {job_id}\n"
            f"Validation: {validation_message}\n"
            f"Dashboard: {dashboard}\n"
            f"Status API: {status_api}"
        )

        if isinstance(terminal_status, dict):
            final_state = str(terminal_status.get("status", "")).upper() or "UNKNOWN"
            message += f"\n\nFinal State: {final_state}"

            output = terminal_status.get("output") if isinstance(terminal_status.get("output"), dict) else {}

            node_report = output.get("node_check_report") if isinstance(output, dict) else None
            if isinstance(node_report, dict) and (node_report.get("missing_before") or []):
                message += f"\n\n{format_node_check_report(node_report)}"

            if final_state in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                runtime_hint = _clip_text(output.get("runtime_hint"))
                error_text = _clip_text(output.get("error"))
                traceback_text = _clip_text(output.get("traceback"))
                comfy_tail = _clip_text(output.get("comfyui_log_tail"))

                if runtime_hint:
                    message += f"\n\nRuntime Hint:\n{runtime_hint}"
                if error_text:
                    message += f"\n\nError:\n{error_text}"
                if traceback_text:
                    message += f"\n\nTraceback:\n{traceback_text}"
                if comfy_tail:
                    message += f"\n\nComfyUI Log Tail:\n{comfy_tail}"
        else:
            message += (
                "\n\nNo terminal job status yet (still running or polling timeout). "
                "Open Status API link for live updates."
            )
        return message
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def run_preflight_check(
    selected_model_label: str,
    selected_model_type: str,
    min_model_size_gb: float,
    search_state: dict[str, dict[str, Any]],
) -> str:
    is_valid, validation_message, _ = validate_selected_model(
        selected_label=selected_model_label,
        search_state=search_state,
        selected_model_type=selected_model_type,
        min_size_gb=min_model_size_gb,
    )
    prefix = "PASS: " if is_valid else "FAIL: "
    return f"{prefix}{validation_message}"


def refresh_startup_status() -> str:
    # Re-evaluate environment/files while the app is running.
    return build_startup_status_markdown()


with gr.Blocks(title="RunPod ComfyUI Command Center") as demo:
    gr.Markdown("## RunPod ComfyUI Command Center")
    gr.Markdown("Trigger your serverless ComfyUI workflow from a local web UI.")
    startup_status = gr.Markdown(build_startup_status_markdown())
    refresh_status_btn = gr.Button("Refresh Startup Check")
    search_state = gr.State({})
    workflow_scan_state = gr.State({})
    auto_models_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            positive = gr.Textbox(
                label="Positive Prompt",
                lines=6,
                placeholder="Describe your scene, style, motion, and camera behavior...",
            )
            negative = gr.Textbox(
                label="Negative Prompt",
                lines=4,
                placeholder="low quality, jitter, artifacts, bad anatomy...",
            )
            frame_length = gr.Slider(
                minimum=1,
                maximum=169,
                step=1,
                value=97,
                label="Frame Length",
            )
            i2v_image = gr.Image(
                label="Image-to-Video Input (Optional)",
                type="filepath",
            )

            with gr.Accordion("Advanced: Model Validation & Search", open=False):
                workflow_file = gr.File(
                    label="Workflow JSON File (optional for scan)",
                    file_types=[".json"],
                    type="filepath",
                )
                workflow_json_text = gr.Textbox(
                    label="Or paste workflow JSON (optional)",
                    lines=6,
                    placeholder="Paste workflow_api.json content here if you do not want to upload a file.",
                )
                scan_workflow_btn = gr.Button("Scan Workflow Models")
                auto_models_btn = gr.Button("Auto-Build input.models From Workflow")
                workflow_models = gr.Dropdown(
                    label="Discovered Workflow Models",
                    choices=[],
                    value=None,
                )
                scan_status = gr.Textbox(
                    label="Workflow Scan Notes",
                    interactive=False,
                    lines=3,
                )
                search_selected_workflow_btn = gr.Button("Search Selected Workflow Model")
                model_query = gr.Textbox(
                    label="Hugging Face Model Search",
                    placeholder="Example: Wan-AI/Wan2.1-T2V-14B or ltx video",
                )
                search_btn = gr.Button("Search Models")
                preflight_btn = gr.Button("Run Pre-Flight Check")
                search_results = gr.Dropdown(
                    label="Select .safetensors / .gguf file",
                    choices=[],
                    value=None,
                )
                selected_model_type = gr.Dropdown(
                    label="Target ComfyUI model type",
                    choices=MODEL_TYPE_CHOICES,
                    value="diffusion_models",
                )
                use_auto_models = gr.Checkbox(
                    label="Use auto-built workflow models in payload",
                    value=True,
                )
                min_size_gb = gr.Slider(
                    label="Minimum Model Size (GB) for pre-flight check",
                    minimum=1,
                    maximum=40,
                    step=1,
                    value=10,
                )
                auto_models_preview = gr.Textbox(
                    label="Auto-built input.models preview",
                    interactive=False,
                    lines=8,
                )
                search_status = gr.Textbox(
                    label="Model Search / Validation Notes",
                    interactive=False,
                    lines=4,
                )

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            status_box = gr.Textbox(
                label="Status",
                lines=12,
                interactive=False,
                placeholder="Job status and errors will appear here...",
            )

    generate_btn.click(
        fn=trigger_job,
        inputs=[
            positive,
            negative,
            frame_length,
            i2v_image,
            workflow_file,
            workflow_json_text,
            search_results,
            selected_model_type,
            min_size_gb,
            search_state,
            use_auto_models,
            auto_models_state,
        ],
        outputs=[status_box],
    )

    search_btn.click(
        fn=search_hf_models,
        inputs=[model_query],
        outputs=[search_results, search_state, search_status],
    )

    scan_workflow_btn.click(
        fn=scan_workflow_for_models,
        inputs=[workflow_file, workflow_json_text],
        outputs=[workflow_models, workflow_scan_state, scan_status],
    )

    auto_models_btn.click(
        fn=auto_build_models_from_workflow,
        inputs=[workflow_file, workflow_json_text],
        outputs=[auto_models_state, auto_models_preview, search_status],
    )

    search_selected_workflow_btn.click(
        fn=search_selected_workflow_model,
        inputs=[workflow_models, workflow_scan_state],
        outputs=[model_query, search_results, search_state, search_status],
    )

    preflight_btn.click(
        fn=run_preflight_check,
        inputs=[search_results, selected_model_type, min_size_gb, search_state],
        outputs=[search_status],
    )

    refresh_status_btn.click(
        fn=refresh_startup_status,
        inputs=[],
        outputs=[startup_status],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
