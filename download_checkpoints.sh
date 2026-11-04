#!/bin/bash
# Downloads the two checkpoints AR-Edit needs into ./checkpoints:
#   checkpoints/Wan2.1-T2V-1.3B/          (UMT5 text encoder, VAE, DiT config)
#   checkpoints/Reward-Forcing-T2V-1.3B/  (rewardforcing.pt, the 4-step generator)
set -e
mkdir -p checkpoints
echo "Downloading Wan2.1-T2V-1.3B..."
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints/Wan2.1-T2V-1.3B
echo "Downloading Reward-Forcing-T2V-1.3B..."
huggingface-cli download JaydenLu666/Reward-Forcing-T2V-1.3B --local-dir checkpoints/Reward-Forcing-T2V-1.3B
echo "Done."
ls -R checkpoints | head -40
