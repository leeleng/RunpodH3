#!/bin/bash
# Point ComfyUI at the models on the Network Volume.
# MODELS_DIR can be changed in the endpoint's environment variables.
MODELS_DIR="${MODELS_DIR:-/runpod-volume/ComfyUI/models}"

cat > /comfyui/extra_model_paths.yaml <<EOF
volume:
  base_path: ${MODELS_DIR}
  checkpoints: checkpoints/
  diffusion_models: diffusion_models/
  unet: unet/
  vae: vae/
  text_encoders: text_encoders/
  clip: clip/
  loras: loras/
EOF

echo "Models dir: ${MODELS_DIR}"
ls "${MODELS_DIR}" || echo "WARNING: models dir not found - check MODELS_DIR"
cd /app && exec python -u handler.py
