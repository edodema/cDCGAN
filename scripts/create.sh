#!/bin/bash
# All labels.
PYTHONPATH=. python src/main.py --dataset "mnist" --channels 1
# Label 2
PYTHONPATH=. python src/main.py --dataset "mnist" --channels 1 --label 2