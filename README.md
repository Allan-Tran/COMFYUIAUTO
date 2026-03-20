# ComfyUI RunPod Serverless Worker

Production-focused RunPod Serverless worker for ComfyUI with on-demand model downloads (zero-idle-cost pattern) and workflow execution through the local ComfyUI API.

## Included

- Base image: `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04`
- System deps: `aria2`, `ffmpeg`, `git-lfs`
- Repos:
  - `comfyanonymous/ComfyUI`
  - `ltdrdata/ComfyUI-Manager`
  - `city96/ComfyUI-GGUF`
  - `kijai/ComfyUI-KJNodes`
  - `kijai/ComfyUI-MediaMixer`
  - `Kosinkadink/ComfyUI-VideoHelperSuite`
- Handler behavior:
  - Downloads models to `/comfyui/ComfyUI/models/{type}` via `aria2c` with 16 connections.
  - Starts ComfyUI headless on `127.0.0.1:8188`.
  - Submits API-format workflow and polls completion.
  - Waits for output settle time to allow Telegram Sender node side effects to complete.

## Payload Format

See `test_input.json` for a complete example:

- `input.workflow`: ComfyUI API workflow JSON object.
- `input.models`: list of model descriptors:

```json
{
  "type": "checkpoints",
  "url": "https://...",
  "name": "model.safetensors"
}
```

The sample workflow is LTX-video-oriented:

- Uses `EmptyLTXVLatentVideo` instead of `EmptyLatentImage`.
- Loads downloaded VAE explicitly through `VAELoader`.
- Uses `VHS_VideoCombine` to produce an MP4 file instead of frame PNGs.
- Includes Telegram placeholders in `TelegramSender` inputs.

If your node pack uses slightly different LTX node IDs, export a known-good API workflow from your ComfyUI UI and replace only `input.workflow` while keeping the same `input.models` format.

## LTX Sampler Note

- The sample keeps `KSampler` for broad compatibility, but some KJNodes versions expose LTX-specific samplers for better temporal behavior.
- If you see temporal artifacts or sampler errors, swap to the LTX-specific sampler node exported from your own ComfyUI environment.

## Environment Variables

- `COMFYUI_PORT` (default: `8188`): local ComfyUI API port.
- `DOWNLOAD_TIMEOUT_SECONDS` (default: `7200`): per-file download timeout for `aria2c`.
- `PROMPT_TIMEOUT_SECONDS` (default: `7200`): max runtime per prompt.
- `DOWNLOAD_WORKERS` (default: `4`): concurrent model file downloads.
- `COMFYUI_LOG_PATH` (default: `/tmp/comfyui.log`): startup/runtime log file path used in startup error messages.

## Telegram Setup

- In your workflow, set `TelegramSender.inputs.bot_token` to your Telegram bot token from BotFather.
- Set `TelegramSender.inputs.chat_id` to your destination chat/channel ID.
- The worker waits for output-settle time before returning success so file writes and Telegram node side effects complete.

## Finding Node IDs for Local Trigger Scripts

Use this when configuring `run_job.py`/`app.py` environment overrides like `POSITIVE_NODE_ID` and `TELEGRAM_NODE_ID`.

1. In ComfyUI, enable Developer Mode options in settings.
2. Export your workflow as API JSON (`Save (API Format)`) to `workflow_api.json`.
3. Open the JSON and inspect top-level keys like `"1"`, `"3"`, and `"99"`: each key is a Node ID.
4. Find the nodes you want to inject by `class_type`:
  - `CLIPTextEncode` for positive and negative prompt text inputs.
  - `EmptyLTXVLatentVideo` (or your latent video node) for frame `length`.
  - `TelegramSender` for `bot_token` and `chat_id`.
5. Set matching values in `.env` if your IDs differ from script defaults.

## Build Context

Use the provided `.dockerignore` to avoid sending unrelated files (for example `Browser/`) during build context upload.

## Docker Hub Commands (Build, Tag, Push)

```bash
docker build -t yourdockerhubuser/comfyui-runpod-worker:v1 .
docker tag yourdockerhubuser/comfyui-runpod-worker:v1 yourdockerhubuser/comfyui-runpod-worker:latest
docker push --all-tags yourdockerhubuser/comfyui-runpod-worker
```
