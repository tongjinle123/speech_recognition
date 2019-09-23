#!/usr/bin/env bash
python main.py train --num_epoch=600 --device_id=1 --ff_size=1024 --batch_size=32 --num_F=2 --num_T=1 --num_worker=32 --F=80
