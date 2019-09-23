#!/usr/bin/env bash
python main.py train --batch_size=32 --ff_size=2048 --num_head=4 --num_epoch=600 --device_id=1 --parser_name='ParserAishell2' --eval_every_iter=30000 --save_every_iter=30000
