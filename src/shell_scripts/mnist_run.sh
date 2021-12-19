#!/bin/sh

for ((i=0; i<5; i++))
do
  echo "Running for fold $i"
  python src/train.py --run mnist --fold "$i" --model "$1"
done