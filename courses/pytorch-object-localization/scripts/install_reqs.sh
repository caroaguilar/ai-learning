#!/usr/bin/env bash

set -o errexit
set -o nounset


# The kuralabs/pytorch-dev image support the execution of startup scripts by placing
# executable scripts in the `/docker-entrypoint-init.d/` directory.

# This script is mounted as a startup script to install extra dependencies and
# clone the dataset used in the Object Localization example

apt-get update \
    && apt-get --yes --no-install-recommends install \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*


cd ws
pip install -r requirements.txt
git clone https://github.com/parth1620/object-localization-dataset.git