#!/bin/bash
# bash run.sh process_conf/multi_window_split_mydataset.yaml
export ROOT=$ROOT
cfg=$1

T=`date +%m%d%H%M`
python robot_data/runner.py --config_path=$cfg