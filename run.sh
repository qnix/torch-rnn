#!/bin/bash

export PATH=/home/qta/torch/install/bin:$PATH

if [ -f train_gpu ]; then
    train_gpu=$(head -1 train_gpu)
else
    train_gpu=$((RANDOM % 2))
    echo $train_gpu > train_gpu
fi

PRINT_EVERY=100
CHECKPOINT_EVERY=1000

function run_train {
    if [ $# -lt 3 ]; then
        echo "run_train <num-layers> <rnn-size> <batch-size>"
        return 0
    fi
    num_layers=$1; shift
    rnn_size=$1; shift
    batch_size=$1; shift
    th train.lua "$@" -input_h5 $input.h5 -input_json data/kaggle.hat/index.json \
       -batch_size $batch_size -seq_length 180 -rnn_size $rnn_size -num_layers $num_layers \
       -dropout 0.25 -gpu $train_gpu \
       -checkpoint_name cv.hat/checkpoint \
       -print_every $PRINT_EVERY -checkpoint_every $CHECKPOINT_EVERY -max_epochs 5 \
       -learning_rate 1.25e-3 -lr_decay_every 5 -lr_decay_factor 0.95
}

if [ "$1" == "prep" ]; then
    input=$2
    if [ -z "$input" ]; then
        echo "no input file provided"
        echo "./run.sh prep <text-input-file>"
        exit 0
    fi
    output=$(basename $input)
    python scripts/preprocess.py --input_txt $input --input_json data/kaggle.hat/index.json \
           --output_h5 $output.h5 --output_json data/kaggle.hat/index.json \
           --val_frac 0.10 --test_frac 0.10

elif [ "$1" == "train" ]; then
    input=$2
    if [ -z "$input" ]; then
        echo "no input file provided"
        exit 0
    fi
    run_train 3 960 200

elif [ "$1" == "train.cont" ]; then
    input=$2
    if [ -z "$input" ]; then
        echo "no input file provided"
        exit 0
    fi
    run_train 3 960 200 -init_from init.960/checkpoint_40000.t7

elif [ "$1" == "train800" ]; then
    input=$2
    if [ -z "$input" ]; then
        echo "no input file provided"
        exit 0
    fi
    run_train 3 800 200

elif [ "$1" == "train800.cont" ]; then
    input=$2
    if [ -z "$input" ]; then
        echo "no input file provided"
        exit 0
    fi
    init_from=init.800.hat/$(ls -1rt init.800.hat | tail -1)
    run_train 3 800 200 -init_from $init_from

elif [ "$1" == "xtrain" ]; then
    input=$2
    train_gpu=$((1 - $train_gpu))
    if [ -z "$input" ]; then
        echo "no input file provided"
        exit 0
    fi
    run_train 5 500 200

elif [ "$1" == "sample" ]; then
    start_text="$(echo $2 | perl -p -e 's/([A-Z])/^\L\1\E/g; s/\s*$//')"
    if [ -z "$start_text" ]; then
        start_text=$(grep -v '^==' output | tail -n "+$((4 + RANDOM % 30))" | head -1 | perl -p -e 's/([A-Z])/^\L\1\E/g; s/\s*$//')
    fi
    checkpoint=$(ls -1rt cv.hat/checkpoint*.t7 | tail -1)
    echo "start_text: $start_text"
    echo "checkpoint: $checkpoint"
    th sample.lua -gpu $((1 - $train_gpu)) -sample 1 -checkpoint $checkpoint \
       -start_text "$start_text" -length 30000 | perl -p -e 's/\^([a-z])/\U\1\E/g' > output
    awk 'length($0) >= 80 && length($0) < 140 && $0 !~ /\yI\y|\y[Mm]e\y|Obama/' output

elif [ "$1" == "sample-cpu" ]; then
    shift
    start_text=$(tail -1 output)
    th sample.lua -gpu -1 -sample 1 -checkpoint $(ls -1rt cv/checkpoint*.t7 | tail -1) \
       -start_text "$start_text" -length 10000 "$@" | \
        awk 'length($0) > 70 && length($0) < 140' | tee output
fi
