#!/bin/bash
#train
CFG="humanact12_rnn"
GPU_IDX=0
python exp_vae_act.py --exp MotionCompass_WAT  --gpu_index $GPU_IDX --cfg $CFG --is_other_act --seed 0 --use_textlabel --use_dir --use_ar --use_ddm --decoder_fuse_w 1.0 --y_w 10 --p_w 10 --lr 2e-3

#test
CFG="humanact12_rnn"
CFG_CLASS=humanact12_act_classifier
GPU_IDX=0
TH=0.01
python eval_vae_act_stats_muti_seed.py --exp MotionCompass_WAT --seed 0 --iter 500 --nk 10 --bs 5 --num_samp 50 --num_seed 5 --stop_fn 5 --cfg $CFG --cfg_classifier $CFG_CLASS --gpu_index $GPU_IDX --threshold $TH --use_textlabel --use_dir --use_ar --use_ddm --decoder_fuse_w 1.0