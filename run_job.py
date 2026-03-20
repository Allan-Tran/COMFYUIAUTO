import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

RUNPOD_API_BASE = "https://api.runpod.ai/v2"
DEFAULT_WORKFLOW_PATH = Path("workflow_api.json")
MODELS_DEFAULTS_PATH = Path("models_defaults.json")


class CliError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a ComfyUI workflow job to a RunPod Serverless endpoint."
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=DEFAULT_WORKFLOW_PATH,
        help="Path to workflow_api.json (or JSON containing input.workflow)",
    )
    parser.add_argument(
        "--telegram-node-id",
        type=str,
        default=os.getenv("TELEGRAM_NODE_ID", "").strip() or None,
        help="Optional explicit TelegramSender node id (example: 99)",
    )
    parser.add_argument(
        "--models-json",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of model objects (type, name, url)",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise CliError(f"Missing required environment variable: {name}")
    return value


def load_workflow(workflow_path: Path) -> dict[str, Any]:
    if not workflow_path.exists():
        raise CliError(f"Workflow file not found: {workflow_path}")

    try:
        raw = json.loads(workflow_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(f"Invalid JSON in {workflow_path}: {exc}") from exc

    if isinstance(raw, dict) and isinstance(raw.get("input"), dict) and isinstance(raw["input"].get("workflow"), dict):
        return raw["input"]["workflow"]

    if isinstance(raw, dict) and raw:
        # Supports direct ComfyUI API workflow export format.
        return raw

    raise CliError("workflow_api.json must be a JSON object or include input.workflow")


def load_models(models_path: Path | None) -> list[dict[str, str]]:
    if models_path is None:
        models_path = MODELS_DEFAULTS_PATH

    if not models_path.exists():
        raise CliError(f"Models JSON file not found: {models_path}")

    try:
        raw = json.loads(models_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(f"Invalid JSON in {models_path}: {exc}") from exc

    if not isinstance(raw, list):
        raise CliError("models JSON must be a list")

    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CliError(f"models[{idx}] must be an object")
        model_type = str(item.get("type", "")).strip()
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        if not model_type or not name or not url:
            raise CliError(f"models[{idx}] requires type, name, and url")
        normalized.append({"type": model_type, "name": name, "url": url})

    return normalized


def resolve_telegram_node_id(workflow: dict[str, Any], explicit_node_id: str | None) -> str:
    if explicit_node_id:
        node_id = str(explicit_node_id)
        if node_id not in workflow:
            raise CliError(f"Telegram node id {node_id!r} was not found in workflow")
        return node_id

    sender_ids = [
        node_id
        for node_id, node in workflow.items()
        if isinstance(node, dict) and str(node.get("class_type", "")).strip() == "TelegramSender"
    ]
    if len(sender_ids) == 1:
        return sender_ids[0]
    if len(sender_ids) > 1:
        raise CliError(
            "Found multiple TelegramSender nodes. Re-run with --telegram-node-id to choose one."
        )

    by_input_keys = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if "bot_token" in inputs and "chat_id" in inputs:
            by_input_keys.append(node_id)

    if len(by_input_keys) == 1:
        return by_input_keys[0]
    if len(by_input_keys) > 1:
        raise CliError(
            "Found multiple nodes with bot_token/chat_id inputs. Re-run with --telegram-node-id."
        )

    raise CliError("Could not auto-detect TelegramSender node. Use --telegram-node-id.")


def inject_telegram_credentials(
    workflow: dict[str, Any], node_id: str, bot_token: str, chat_id: str
) -> None:
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        raise CliError(f"Workflow node {node_id!r} is not an object")

    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
        node["inputs"] = inputs

    inputs["bot_token"] = bot_token
    inputs["chat_id"] = chat_id


def submit_runpod_job(endpoint_id: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    run_url = f"{RUNPOD_API_BASE}/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(run_url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise CliError(f"Request to RunPod failed: {exc}") from exc

    if response.status_code >= 400:
        body = response.text.strip()
        snippet = body[:1500] if body else "<empty body>"
        raise CliError(f"RunPod API error {response.status_code}: {snippet}")

    try:
        data = response.json()
    except ValueError as exc:
        raise CliError(f"RunPod API returned non-JSON response: {response.text[:500]}") from exc

    return data


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        api_key = require_env("RUNPOD_API_KEY")
        endpoint_id = require_env("RUNPOD_ENDPOINT_ID")
        bot_token = require_env("TELEGRAM_BOT_TOKEN")
        chat_id = require_env("TELEGRAM_CHAT_ID")

        workflow = load_workflow(args.workflow)
        telegram_node_id = resolve_telegram_node_id(workflow, args.telegram_node_id)
        inject_telegram_credentials(workflow, telegram_node_id, bot_token, chat_id)

        models = load_models(args.models_json)

        payload = {
            "input": {
                "workflow": workflow,
                "models": models,
            }
        }

        result = submit_runpod_job(endpoint_id=endpoint_id, api_key=api_key, payload=payload)

        job_id = result.get("id") or result.get("jobId") or result.get("requestId")
        if not job_id:
            raise CliError(f"RunPod response did not include a job id: {result}")

        print(f"Submitted successfully. Job ID: {job_id}")
        print(
            f"RunPod Dashboard: https://www.runpod.io/console/serverless/user/endpoints/{endpoint_id}"
        )
        print(f"RunPod API Status: {RUNPOD_API_BASE}/{endpoint_id}/status/{job_id}")
        return 0

    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
