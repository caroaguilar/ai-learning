#!/usr/bin/env bash

set -o errexit
set -o nounset


echo "Pulling PyTorch Dev image..."
docker pull kuralabs/pytorch-dev:latest

echo "Starting new Container ..."
docker run \
    --rm \
    --gpus all \
    --tty \
    --interactive \
    --name pytorch-basics \
    --volume $(pwd)/ws:/home/pytorch/ws \
    kuralabs/pytorch-dev:latest bash


