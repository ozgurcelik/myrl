#!/bin/bash

WORK_DIR=$(pwd)
CACHE_DIR=$WORK_DIR/.cache
mkdir -p $CACHE_DIR

HF_HOME="$CACHE_DIR/huggingface" python grpo/main.py