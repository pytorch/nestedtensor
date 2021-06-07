#!/usr/bin/env bash
set -x
set -e

cd /mnt/mydata/scripts
apt update
apt-get -y install git wget
export PARAMETERS_PYTHON_VERSION="3.8"
.circleci/unittest/linux/scripts/setup_env.sh
.circleci/unittest/linux/scripts/install.sh
.circleci/unittest/linux/scripts/run_test.sh
