#!/bin/bash
module load conda
conda activate .cvenv
module load ffmpeg
exec python "$@"