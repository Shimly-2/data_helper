#!/bin/bash
# [rt1 settings]
export ROOT=$ROOT
cfg=$1

T=`date +%m%d%H%M`
python robot_data/runner.py --config_path=$cfg