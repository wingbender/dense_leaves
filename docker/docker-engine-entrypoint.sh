#!/bin/bash

. /home/user/miniconda/etc/profile.d/conda.sh

conda activate ${PROJECT_VENV}

exec "$@"

