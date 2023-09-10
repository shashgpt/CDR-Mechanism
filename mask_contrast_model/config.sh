#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=UNLIMITED
#SBATCH --nodelist=legolas
#SBATCH --output=./assets/training_logs/%x.out
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
--hidden_units_contrast_embedder "${15}" \
--generate_explanation_for_one_instance "${16}"