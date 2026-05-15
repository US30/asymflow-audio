#!/usr/bin/env bash
# Download LJSpeech-1.1 dataset (~2.6 GB, CC0 license).
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
ARCHIVE="$DATA_DIR/LJSpeech-1.1.tar.bz2"

echo "Downloading LJSpeech-1.1 (~2.6 GB)..."
curl -L "$URL" -o "$ARCHIVE"

echo "Extracting..."
tar -xjf "$ARCHIVE" -C "$DATA_DIR"
rm "$ARCHIVE"

echo "Done. Data in: $DATA_DIR/LJSpeech-1.1/"
echo "  wavs/   : $(ls "$DATA_DIR/LJSpeech-1.1/wavs/" | wc -l) .wav files"
