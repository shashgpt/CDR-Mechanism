#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=UNLIMITED
#SBATCH --nodelist=legolas
#SBATCH --output=./assets/training_logs/%x.out

#When running locally on PC
# source /home/guptashas/anaconda/bin/activate env_python_3.9_tensorflow
# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
# /home/guptashas/.conda/envs/env_python_3.9_tensorflow/bin/python main.py \
# --asset_name "lstm_lstm_mask_model-MASK_OUTPUT_CORRECTED-TRAINED_ON_SERVER" \
# --model_name "lstm_lstm_mask" \
# --seed_value 11 \
# --dataset_name "Covid-19_tweets" \
# --dataset_path "datasets/Covid-19_tweets/raw_dataset.pickle" \
# --fine_tune_embedding_model False \
# --optimizer "adam" \
# --learning_rate 3e-5 \
# --mini_batch_size 50 \
# --train_epochs 1 \
# --dropout 0.4 \
# --lime_no_of_samples 1000 \
# --hidden_units_classifier 128 \
# --hidden_units_mask_embedder 128 \
# --generate_explanation_for_one_instance True

# #When running on pippin or luthin servers
# export CUDA_VISIBLE_DEVICES=$1
# source /home/guptashas/anaconda/bin/activate env_python_3.9_tensorflow
# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
# /home/guptashas/.conda/envs/env_python_3.9_tensorflow/bin/python main.py \
# --asset_name $2 \
# --model_name $3 \
# --seed_value $4 \
# --dataset_name $5 \
# --dataset_path $6 \
# --fine_tune_embedding_model $7 \
# --optimizer $8 \
# --learning_rate $9 \
# --mini_batch_size ${10} \
# --train_epochs "${11}" \
# --dropout "${12}" \
# --lime_no_of_samples "${13}" \
# --hidden_units_classifier "${14}" \
# --hidden_units_mask_embedder "${15}" \
# --generate_explanation_for_one_instance "${16}"
# exec bash

source /home/guptashas/anaconda/bin/activate env_python_3.9_tensorflow
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
python main.py \
--asset_name $1 \
--model_name $2 \
--seed_value $3 \
--dataset_name $4 \
--dataset_path $5 \
--fine_tune_embedding_model $6 \
--optimizer $7 \
--learning_rate $8 \
--mini_batch_size $9 \
--train_epochs "${10}" \
--dropout "${11}" \
--lime_no_of_samples "${12}" \
--hidden_units_classifier "${13}" \
--hidden_units_mask_embedder "${14}" \
--generate_explanation_for_one_instance "${15}"

# exec 3>&1 4>&2
# trap 'exec 2>&4 1>&3' 0 1 2 3
# exec 1>"${17}"/$2.out 2>&1
# (/home/guptashas/.conda/envs/env_python_3.9_tensorflow/bin/python main.py \
# --asset_name $1 \
# --model_name $2 \
# --seed_value $3 \
# --dataset_name $4 \
# --dataset_path $5 \
# --fine_tune_embedding_model $6 \
# --optimizer $7 \
# --learning_rate $8 \
# --mini_batch_size $9 \
# --train_epochs "${10}" \
# --dropout "${11}" \
# --lime_no_of_samples "${12}" \
# --hidden_units_classifier "${13}" \
# --hidden_units_mask_embedder "${14}" \
# --hidden_units_contrast_embedder "${15}" \
# --generate_explanation_for_one_instance "${16}" || exec bash)

# export CUDA_VISIBLE_DEVICES=$1
# export PYTHONPATH=/home/guptashas/PhD_experiments/CDR-mechanism/mask_contrast_model
# (cd /home/guptashas/PhD_experiments/CDR-mechanism/mask_contrast_model && /home/guptashas/.conda/envs/env_python_3.8_tensorflow/bin/python main.py \
# --asset_name $2 \
# --model_name $3 \
# --seed_value $4 \
# --dataset_name $5 \
# --fine_tune_embedding_model $6 \
# --optimizer $7 \
# --learning_rate $8 \
# --mini_batch_size $9 \
# --train_epochs $10 \
# --dropout $11 \
# --lime_no_of_samples $12 \
# --hidden_units_classifier $13 \
# --hidden_units_mask_embedder $14 )
# # exec bash