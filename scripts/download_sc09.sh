#!/usr/bin/env bash
# Download SC09: Speech Commands v0.02, extract only digit classes.
set -euo pipefail

DATA_DIR="${1:-data/sc09}"
mkdir -p "$DATA_DIR"

URL="https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
TMP="$(mktemp -d)"
ARCHIVE="$TMP/sc.tar.gz"

echo "Downloading Speech Commands v0.02 (~2.4GB)..."
curl -L "$URL" -o "$ARCHIVE"

echo "Extracting digit classes..."
DIGITS=(zero one two three four five six seven eight nine)
for digit in "${DIGITS[@]}"; do
    tar -xzf "$ARCHIVE" -C "$DATA_DIR" "./$digit" 2>/dev/null || \
    tar -xzf "$ARCHIVE" -C "$DATA_DIR" "$digit"
done

rm -rf "$TMP"
echo "Done. Data in: $DATA_DIR"
ls "$DATA_DIR"
