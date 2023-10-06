#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu-dev-1080
#SBATCH --time=UNLIMITED
#SBATCH --nodelist=thorin-1
#SBATCH --output=./assets/training_logs/%x.out

# When testing one configuration (PROTOTYPING)
source /home/guptashas/anaconda/bin/activate env_python_3.9_tensorflow
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
python main.py \
--asset_name "gru_FE_model-MODEL_MODIFIED_FOR_PERCY_CALCULATION-TRAINED_ON_SERVER" \
--model_name "gru_FE" \
--seed_value 11 \
--dataset_name "Covid-19_tweets" \
--dataset_path "datasets/Covid-19_tweets/raw_dataset.pickle" \
--fine_tune_embedding_model "False" \
--optimizer "adam" \
--learning_rate 3e-5 \
--mini_batch_size 50 \
--train_epochs 1 \
--dropout 0.4 \
--lime_no_of_samples 1000 \
--hidden_units_classifier 128 \
--generate_explanation_for_one_instance "True" \
--train_model "True" \
--evaluate_model "True" \
--generate_explanation "True" \
--prototyping "True"

# # When being trained on server (PROTOTYPING)
# time_stamp=$(date)
# rm "$1.out"
# exec 3>&1 4>&2
# trap 'exec 2>&4 1>&3' 0 1 2 3
# exec 1>$1.out 2>&1
# source /home/guptashas/anaconda/bin/activate env_python_3.9_tensorflow
# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
# (python main.py \
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
# --generate_explanation_for_one_instance "${14}" \
# --train_model "${15}" \
# --evaluate_model "${16}" \
# --generate_explanation "${17}" \
# --prototyping "False" || exec bash)

# # When using slurm
# source /home/guptashas/anaconda/bin/activate env_python_3.9_tensorflow
# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
# python main.py \
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
# --generate_explanation_for_one_instance "${14}" \
# --train_model "${15}" \
# --evaluate_model "${16}" \
# --generate_explanation "${17}"