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
    --name pytorch-dev \
    --volume $(pwd)/ws:/home/pytorch/ws \
    --volume $(pwd)/scripts:/docker-entrypoint-init.d/ \
    --env ADJUST_USER_UID=$(id -u) \
    --env ADJUST_USER_GID=$(id -g) \
    kuralabs/pytorch-dev:latest bash