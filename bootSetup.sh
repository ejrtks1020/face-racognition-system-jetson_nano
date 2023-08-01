#!/bin/bash

sudo mount /dev/sda1 /usr/local
bash Desktop/installSwapfile.sh -d /usr/local -s 4 -a F
sudo jetson_clocks
sudo bash -c 'echo 100 > /sys/devices/pwm-fan/target_pwm'

export LD_PRELOAD=/usr/local/lib/python3.6/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
cd /home/jetson/Desktop/ailab-face-cam-server
python3 cam_main_multi.py
