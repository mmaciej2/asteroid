#!/bin/bash

set -e

source activate asteroid-dev
module load cuda11.6/toolkit

$python_path $@
