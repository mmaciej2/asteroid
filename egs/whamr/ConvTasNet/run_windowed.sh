#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
storage_dir=

# If you start from the sphere files, specify the path to the directory and start from stage 0
sphere_dir=  # Directory containing sphere files
# If you already have wsj0 wav files, specify the path to the directory here and start from stage 1
wsj0_wav_dir=
# If you already have the WHAM mixtures, specify the path to the directory here and start from stage 2
whamr_wav_dir=/expscratch/mmaciejewski/datasets/whamr
# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=4  # Controls from which stage to start
tag="sep_reverb_16kmin_windowed"  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Data
task=sep_reverb  # Specify the task here (sep_clean, sep_noisy, sep_reverb, sep_reverb_noisy)
sample_rate=16000
mode=min
nondefault_src=  # If you want to train a network with 3 output streams for example.
data_dir=data  # Local data directory (No disk space needed)

# Training
batch_size=4
num_workers=4
lr=0.001
epochs=100
weight_decay=0.00001

# Architecture
fb_name=windowed_free
n_filters=512
kernel_size=32
stride=16

n_blocks=8
n_repeats=3
mask_nonlinear=relu

# Evaluation
eval_use_gpu=1

. utils/parse_options.sh

sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k/$mode
dumpdir=data/$suffix  # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
test_dir=$dumpdir/tt

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
#  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
  qsub -cwd -S /bin/bash -r no -m eas -M mmaciej2@jhu.edu \
    -q gpu.q@@1080 -l gpu=1,mem_free=10G,h_rt=72:00:00,num_proc=1 \
    -N "train_${tag}" \
    -v python_path=$python_path \
    -j y -o logs/train_${tag}.log \
    qsub_gpu.sh train.py \
		--train_dir $train_dir \
		--valid_dir $valid_dir \
		--task $task \
		--sample_rate $sample_rate \
		--lr $lr \
		--weight_decay $weight_decay \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--fb_name $fb_name \
		--n_filters $n_filters \
		--kernel_size $kernel_size \
		--stride $stride \
		--n_blocks $n_blocks \
		--n_repeats $n_repeats \
		--mask_act $mask_nonlinear \
		--exp_dir ${expdir}
#		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
#	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
#	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
  qsub -cwd -S /bin/bash -r no -m eas -M mmaciej2@jhu.edu \
    -q gpu.q@@1080 -l gpu=1,mem_free=10G,h_rt=1:00:00,num_proc=1 \
    -hold_jid "train_${tag}" -N "eval_${tag}" \
    -v python_path=$python_path \
    -j y -o logs/eval_${tag}.log \
    qsub_gpu.sh eval.py \
		--task $task \
		--test_dir $test_dir \
		--use_gpu $eval_use_gpu \
		--exp_dir ${expdir}
#		--exp_dir ${expdir} | tee logs/eval_${tag}.log
#	cp logs/eval_${tag}.log $expdir/eval.log
fi
