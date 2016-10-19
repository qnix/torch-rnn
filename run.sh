#!/bin/bash -x

train_gpu=0

PRINT_EVERY=100
CHECKPOINT_EVERY=1000

function run_train {
    if [ $# -lt 2 ]; then
        echo "run_train <rnn-size> <batch-size>"
        return 0
    fi
    rnn_size=$1; shift
    batch_size=$1; shift
    th train.lua "$@" -input_h5 $input.h5 -input_json data/index.json \
       -batch_size $batch_size -seq_length 180 -rnn_size $rnn_size -num_layers 3 \
       -dropout 0.25 -gpu $train_gpu \
       -print_every $PRINT_EVERY -checkpoint_every $CHECKPOINT_EVERY -max_epochs 5 \
       -learning_rate 1.25e-3 -lr_decay_every 2 -lr_decay_factor 0.8
}

if [ "$1" == "prep" ]; then
    input=$2
    if [ -z "$input" ]; then
        echo "no input file provided"
        echo "./run.sh prep <text-input-file>"
        exit 0
    fi
    output=$(basename $input)
    python scripts/preprocess.py --input_txt $input --input_json data/index.json \
           --output_h5 $output.h5 --output_json data/index.json \
           --val_frac 0.10 --test_frac 0.10

elif [ "$1" == "train" ]; then
    input=$2
    if [ -z "$input" ]; then
        echo "no input file provided"
        exit 0
    fi
    run_train 960 200

elif [ "$1" == "train.cont" ]; then
    input=$2
    if [ -z "$input" ]; then
        echo "no input file provided"
        exit 0
    fi
    run_train 960 200 -init_from init.960/checkpoint_40000.t7

elif [ "$1" == "train800" ]; then
    input=$2
    if [ -z "$input" ]; then
        echo "no input file provided"
        exit 0
    fi
    run_train 800 200

elif [ "$1" == "train800.cont" ]; then
    input=$2
    if [ -z "$input" ]; then
        echo "no input file provided"
        exit 0
    fi
    init_from=init.800/$(ls -1rt init.800 | tail -1)
    run_train 800 200 -init_from $init_from

elif [ "$1" == "sample" ]; then
    start_text="$2"
    if [ -z "$start_text" ]; then
        start_text=$(tail -1 output | cut -c 1-30)
    fi
    checkpoint=$(ls -1rt cv/checkpoint*.t7 | tail -1)
    echo "start_text: $start_text"
    echo "checkpoint: $checkpoint"
    th sample.lua -gpu $((1 - $train_gpu)) -sample 1 -checkpoint $checkpoint \
       -start_text "$start_text" -length 20000 > output
    awk 'length($0) > 80 && length($0) < 140' output
    head -2 output

elif [ "$1" == "sample-cpu" ]; then
    shift
    start_text=$(tail -1 output | cut -c 1-30)
    th sample.lua -gpu -1 -sample 1 -checkpoint $(ls -1rt cv/checkpoint*.t7 | tail -1) \
       -start_text "$start_text" -length 10000 "$@" | \
        awk 'length($0) > 70 && length($0) < 140' | tee output
fi
