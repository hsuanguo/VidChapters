#! /usr/bin/env bash

set -e
set -u
set -o pipefail

cd /workspace/VidChapters

# check file in /input, there should be only one file
input_video_file=$(ls /input)
echo "input_video_file: ${input_video_file}"

# input model
input_model=$(ls /input_model)
echo "input_model: ${input_model}"

# input asr
input_asr=$(ls /input_asr)
echo "input_asr: ${input_asr}"

python3 demo_vid2seq.py --load="${input_model}" --video_example="${input_video_file}" --asr_example "${input_asr}" --combine_datasets chapters

