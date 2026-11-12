FROM runpod/pytorch:2.1.0-cu1281-torch290-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /comfyui
WORKDIR /comfyui
RUN pip install --no-cache-dir --ignore-installed cryptography && pip install --no-cache-dir -r requirements.txt runpod requests

COPY handler.py workflow_api.json start.sh /app/
RUN chmod +x /app/start.sh
ENTRYPOINT []
CMD ["bash", "/app/start.sh"]
