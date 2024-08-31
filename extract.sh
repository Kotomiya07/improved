#!/bin/sh
export PYTHONPATH=$(pwd):$PYTHONPATH

CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

DATASET=$1
SIZE=$2

echo $DATASET $SIZE


if [[ $DATASET == cifar10 ]]; then
    python3 extract_features.py --dataset cifar10 --datadir data/cifar-10/ --image_size 32 --vae ema

elif [[ $DATASET == coco ]]; then
    python3 extract_features.py --dataset coco --datadir data/coco/ --image_size $SIZE --vae ema

elif [[ $DATASET == celeba_256 ]]; then
    python3 extract_features.py --dataset celeba_256 --datadir data/celeba/celeba-lmdb/ --image_size 256 --vae ema

elif [[ $DATASET == lsun ]]; then
    python3 extract_features.py --dataset lsun --datadir data/lsun/ --image_size $SIZE --vae ema
fi