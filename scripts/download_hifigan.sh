#!/usr/bin/env bash
# Download pretrained HiFi-GAN UNIVERSAL_V1 vocoder and clone the repo.
# Source: https://github.com/jik876/hifi-gan
# Checkpoint: universal V1 (works across multiple speakers/conditions).
set -euo pipefail

HIFIGAN_DIR="third_party/hifigan"
DATA_DIR="data/hifigan"
mkdir -p "$DATA_DIR"

# Clone HiFi-GAN repo (needed for inference code)
if [ ! -d "$HIFIGAN_DIR" ]; then
    echo "Cloning jik876/hifi-gan..."
    git clone https://github.com/jik876/hifi-gan.git "$HIFIGAN_DIR"
else
    echo "HiFi-GAN repo already at $HIFIGAN_DIR"
fi

# Download pretrained UNIVERSAL_V1 checkpoint (~50 MB)
echo "Downloading UNIVERSAL_V1 checkpoint..."
curl -L \
  "https://drive.google.com/uc?export=download&id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW" \
  -o "$DATA_DIR/generator_universal.pth" || {
    echo ""
    echo "WARNING: Google Drive auto-download may have failed (quota/redirect issue)."
    echo "Manually download UNIVERSAL_V1 from:"
    echo "  https://github.com/jik876/hifi-gan#pretrained-model"
    echo "and place generator checkpoint as: $DATA_DIR/generator_universal.pth"
    echo "and config as: $DATA_DIR/config_v1.json"
}

# Download config
CONFIG_URL="https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json"
echo "Downloading config_v1.json..."
curl -L "$CONFIG_URL" -o "$DATA_DIR/config_v1.json"

echo ""
echo "HiFi-GAN setup complete."
echo "  Repo:       $HIFIGAN_DIR/"
echo "  Checkpoint: $DATA_DIR/generator_universal.pth"
echo "  Config:     $DATA_DIR/config_v1.json"
echo ""
echo "If the checkpoint download failed, see manual instructions above."
echo "Alternatively, the code will fall back to Griffin-Lim (lower quality)."
