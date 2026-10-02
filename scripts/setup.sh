#!/bin/sh
# Fetch everything that is not stored in this repository: the Pain Axis release (for the
# 7B pain vector and the sentence sets) at a pinned commit, and the 7B weights. The 1.5B
# weights are fetched by mlx-lm from Hugging Face on first use.
set -e
cd "$(dirname "$0")/.."

if [ ! -e external/Pain-axis ]; then
  mkdir -p external
  git clone https://github.com/valen-research/Pain-axis external/Pain-axis
  git -C external/Pain-axis checkout 4d75cd90e206ea962f7a9101e65c85efea56723b
fi

M=models/Qwen2.5-7B-Instruct-4bit
REV=c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed
if [ ! -e "$M/model.safetensors" ]; then
  mkdir -p "$M"
  for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json model.safetensors; do
    until curl -sSL -C - --speed-limit 100000 --speed-time 30 -o "$M/$f" \
        "https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit/resolve/$REV/$f"; do
      echo "retrying $f"
    done
  done
fi
echo "setup done"
