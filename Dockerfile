FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Latest ComfyUI (the MiniMax H3 nodes are in core, so no custom nodes are needed)
RUN git clone https://github.com/comfyanonymous/ComfyUI /comfyui
WORKDIR /comfyui
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir runpod requests

COPY handler.py workflow_api.json start.sh /app/
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
