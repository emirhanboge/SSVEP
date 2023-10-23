#!/bin/bash

if [ -z "$1" ]; then
  echo "Please provide the image path."
  exit 1
fi

DATASET="morph2"
STATE_DICT_PATH="../morph2/morph-ce__seed1/best_model.pt"

if [ "$2" == "True" ]; then
  python resnet.py --dataset $DATASET --image_path $1 --state_dict_path $STATE_DICT_PATH --enable_shap
else
  python resnet.py --dataset $DATASET --image_path $1 --state_dict_path $STATE_DICT_PATH
fi

