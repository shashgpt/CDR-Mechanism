#!/bin/bash

mkdir -p assets/training_logs/

declare -a dnn_models=("rnn" "birnn" "gru" "bigru" "lstm" "bilstm")
declare -i config_count=0

for dnn_model in ${dnn_models[@]}
do
    asset_name="${dnn_model}_FE_model-MODEL_MODIFIED_FOR_PERCY_CALCULATION-TRAINED_ON_SERVER"
    model_name="${dnn_model}_FE"
    seed_value=11
    dataset_name="Covid-19_tweets"
    dataset_path="datasets/Covid-19_tweets/raw_dataset.pickle"
    fine_tune_embedding_model="False"
    optimizer="adam"
    learning_rate=3e-5
    mini_batch_size=50
    train_epochs=1
    dropout=0.4
    lime_no_of_samples=1000
    hidden_units_classifier=128
    generate_explanation_for_one_instance="False"
    train_model="False"
    evaluate_model="True"
    generate_explanation="True"

    # Running on hpc server
    screen_name=$model_name
    screen -S ${screen_name} -d -m bash config.sh $asset_name $model_name $seed_value $dataset_name $dataset_path $fine_tune_embedding_model $optimizer $learning_rate $mini_batch_size $train_epochs $dropout $lime_no_of_samples $hidden_units_classifier $generate_explanation_for_one_instance $train_model $evaluate_model $generate_explanation
    sleep 5 #so that the process starts (detects the gpu with highest mem and starts executing on it) before another process

    # # Running via slurm queue
    # sbatch --job-name $asset_name config.sh $asset_name $model_name $seed_value $dataset_name $dataset_path $fine_tune_embedding_model $optimizer $learning_rate $mini_batch_size $train_epochs $dropout $lime_no_of_samples $hidden_units_classifier $generate_explanation_for_one_instance $train_model $evaluate_model $generate_explanation
    
    echo $asset_name
    config_count=$(( config_count+1 ))
done