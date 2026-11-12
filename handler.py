"""RunPod serverless handler: MiniMax H3 reference-to-video via ComfyUI.

Job input (all optional except prompt and images):
{
  "input": {
    "prompt": "subject_definitions: ...",
    "images": [ {"name": "scene.png", "data": "<base64 or https URL>"}, ... ],   # up to 9, in <Picture N> order
    "audio":  {"name": "voice.wav", "data": "<base64 or https URL>"},            # optional reference voice
    "duration": 13,                      # seconds
    "seed": 123,                         # optional, random if missing
    "aspect_ratio": "9:16 (Portrait Widescreen)",
    "megapixels": 0.4,
    "turbo": true,                       # true = 4-step lightning LoRA (fast), false = 20 steps
    "nsfw_lora": false                   # enables the HMNSFW LoRA switch in the workflow (off by default)
  }
}
"""
import base64
import copy
import json
import os
import random
import subprocess
import time
import urllib.request
import uuid

import runpod

COMFY = "http://127.0.0.1:8188"
COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")
INPUT_DIR = os.path.join(COMFY_DIR, "input")
OUTPUT_DIR = os.path.join(COMFY_DIR, "output")
WORKFLOW = json.load(open(os.path.join(os.path.dirname(__file__), "workflow_api.json")))

# Node ids in workflow_api.json
IMAGE_NODES = ["137", "139", "149", "150", "151"]  # LoadImage nodes feeding ref_image_0..4
REF_NODE = "136"
AUDIO_NODE = "153"
PROMPT_NODE = "138"
DURATION_NODE = "132"
SEED_NODE = "129"
RES_NODE = "115"
TURBO_NODE = "146"
NSFW_NODE = "148"


def start_comfy():
    subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"],
        cwd=COMFY_DIR,
    )
    for _ in range(600):
        try:
            urllib.request.urlopen(f"{COMFY}/system_stats", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("ComfyUI did not start")


def save_file(item, default_name):
    name = os.path.basename(item.get("name") or default_name)
    name = f"{uuid.uuid4().hex[:8]}_{name}"
    data = item["data"]
    path = os.path.join(INPUT_DIR, name)
    if data.startswith("http"):
        urllib.request.urlretrieve(data, path)
    else:
        if "," in data[:100]:
            data = data.split(",", 1)[1]
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))
    return name


def build_workflow(inp):
    wf = copy.deepcopy(WORKFLOW)
    wf[PROMPT_NODE]["inputs"]["value"] = inp["prompt"]

    images = inp.get("images") or []
    if not images:
        raise ValueError("At least one image is required in 'images'")
    if len(images) > 9:
        raise ValueError("Maximum 9 reference images")
    # Fill existing LoadImage nodes; extend with new ones if more than 5; remove unused ones.
    for i, item in enumerate(images):
        fname = save_file(item, f"ref{i}.png")
        if i < len(IMAGE_NODES):
            node_id = IMAGE_NODES[i]
        else:
            node_id = str(900 + i)
            wf[node_id] = {"inputs": {"image": fname}, "class_type": "LoadImage"}
        wf[node_id]["inputs"]["image"] = fname
        wf[REF_NODE]["inputs"][f"ref_images.ref_image_{i}"] = [node_id, 0]
    for i in range(len(images), len(IMAGE_NODES)):
        wf[REF_NODE]["inputs"].pop(f"ref_images.ref_image_{i}", None)
        wf.pop(IMAGE_NODES[i], None)

    audio = inp.get("audio")
    if audio:
        wf[AUDIO_NODE]["inputs"]["audio"] = save_file(audio, "voice.wav")
    else:
        wf[REF_NODE]["inputs"].pop("ref_audios.ref_audio_0", None)
        wf.pop(AUDIO_NODE, None)

    wf[DURATION_NODE]["inputs"]["value"] = float(inp.get("duration", 13))
    wf[SEED_NODE]["inputs"]["noise_seed"] = int(inp.get("seed", random.randint(0, 2**50)))
    wf[RES_NODE]["inputs"]["aspect_ratio"] = inp.get("aspect_ratio", wf[RES_NODE]["inputs"]["aspect_ratio"])
    wf[RES_NODE]["inputs"]["megapixels"] = float(inp.get("megapixels", wf[RES_NODE]["inputs"]["megapixels"]))
    wf[TURBO_NODE]["inputs"]["value"] = bool(inp.get("turbo", True))
    # In the workflow the switch node 141 picks the NSFW-LoRA model when TURBO is true.
    # Keep behaviour explicit: turbo=true uses the lightning+realism(+nsfw) chain.
    # nsfw_lora=false keeps the HMNSFW LoRA at strength 0 so it has no effect.
    wf[NSFW_NODE]["inputs"]["strength_model"] = 1.0 if inp.get("nsfw_lora") else 0.0
    return wf


def handler(job):
    inp = job["input"]
    wf = build_workflow(inp)
    client_id = uuid.uuid4().hex
    req = urllib.request.Request(
        f"{COMFY}/prompt",
        data=json.dumps({"prompt": wf, "client_id": client_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        prompt_id = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}

    timeout = int(inp.get("timeout", 1500))
    start = time.time()
    while time.time() - start < timeout:
        hist = json.load(urllib.request.urlopen(f"{COMFY}/history/{prompt_id}"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                return {"error": json.dumps(entry["status"].get("messages", [])[-3:])}
            for node_out in entry.get("outputs", {}).values():
                for key in ("images", "videos", "gifs"):
                    for f in node_out.get(key, []):
                        if f["filename"].endswith(".mp4"):
                            path = os.path.join(OUTPUT_DIR, f.get("subfolder", ""), f["filename"])
                            with open(path, "rb") as fh:
                                b64 = base64.b64encode(fh.read()).decode()
                            return {"video_base64": b64, "filename": f["filename"]}
            return {"error": "Workflow finished but no mp4 was found", "outputs": entry.get("outputs")}
        time.sleep(2)
    return {"error": "timeout"}


start_comfy()
runpod.serverless.start({"handler": handler})
