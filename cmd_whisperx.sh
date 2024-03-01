#! /usr/bin/env bash

set -e
set -u
set -o pipefail

cd /workspace/VidChapters

# check file in /input, there should be only one file
input_video_file=$(ls /input)
echo "input_video_file: ${input_video_file}"

# run ASR
python3 demo_asr.py --video_example /input/$input_video_file --combine_datasets chapters
