#!/bin/bash

# declare -a masks=("gru" "bigru" "lstm" "bilstm")
# declare -a dnn_models=("gru" "bigru" "lstm" "bilstm")

# num_configurations=$(( ${#masks[@]}*${#dnn_models[@]} ))
# end_value=$(( $num_configurations-1 ))

# declare -a cuda_devices=(0 1 2 3 4 5 6 7)

# # declare -a cpus
# # for i in $(seq 0 $end_value)
# # do
# #     cpus+=($i)
# # done

# declare -i config_count=0
# for mask in ${masks[@]}
# do  
#     for dnn_model in ${dnn_models[@]}
#     do
#         asset_name="${dnn_model}_${mask}_mask_model-MASK_OUTPUT_CORRECTED-TRAINED_ON_SERVER"
#         model_name="${dnn_model}_${mask}_mask"
#         seed_value=11
#         dataset_name="Covid-19_tweets"
#         dataset_path="datasets/Covid-19_tweets/raw_dataset.pickle"
#         fine_tune_embedding_model=False
#         optimizer="adam"
#         learning_rate=3e-5
#         mini_batch_size=50
#         train_epochs=200
#         dropout=0.4
#         lime_no_of_samples=1000
#         hidden_units_classifier=128
#         hidden_units_mask_embedder=128
#         generate_explanation_for_one_instance=False

#         # cpu=${cpus[$config_count]}
#         multiple=`expr $config_count / ${#cuda_devices[@]}`
#         cuda_device_no=$(( $config_count-${#cuda_devices[@]}*$multiple ))

#         screen_name=$model_name
#         screen -S ${screen_name} -d -m bash config.sh $cuda_device_no $asset_name $model_name $seed_value $dataset_name $dataset_path $fine_tune_embedding_model $optimizer $learning_rate $mini_batch_size $train_epochs $dropout $lime_no_of_samples $hidden_units_classifier $hidden_units_mask_embedder $generate_explanation_for_one_instance

#         config_count=$(( config_count+1 ))
#     done
# done

mkdir -p assets/training_logs/

declare -a masks=("gru" "bigru" "lstm" "bilstm")
declare -a dnn_models=("gru" "bigru" "lstm" "bilstm")

declare -i config_count=0
for mask in ${masks[@]}
do  
    for dnn_model in ${dnn_models[@]}
    do
        asset_name="${dnn_model}_${mask}_mask_model-MASK_OUTPUT_CORRECTED-TRAINED_ON_SERVER"
        model_name="${dnn_model}_${mask}_mask"
        seed_value=11
        dataset_name="Covid-19_tweets"
        dataset_path="datasets/Covid-19_tweets/raw_dataset.pickle"
        fine_tune_embedding_model=False
        optimizer="adam"
        learning_rate=3e-5
        mini_batch_size=50
        train_epochs=200
        dropout=0.4
        lime_no_of_samples=1000
        hidden_units_classifier=128
        hidden_units_mask_embedder=128
        generate_explanation_for_one_instance=False

        # sbatch --job-name=$model_name --partition=gpu --gres=gpu:a100:1 --mem=50G --nodelist=legolas --time=UNLIMITED --output=./log_$run/%x.out config.sh $asset_name $model_name $seed_value $dataset_name $dataset_path $fine_tune_embedding_model $optimizer $learning_rate $mini_batch_size $train_epochs $dropout $lime_no_of_samples $hidden_units_classifier $hidden_units_mask_embedder $hidden_units_contrast_embedder

        # screen_name=$model_name
        # screen -S ${screen_name} -d -m bash config.sh $asset_name $model_name $seed_value $dataset_name $dataset_path $fine_tune_embedding_model $optimizer $learning_rate $mini_batch_size $train_epochs $dropout $lime_no_of_samples $hidden_units_classifier $hidden_units_mask_embedder $generate_explanation_for_one_instance
        # sleep 5 #so that the process starts (detects the gpu with highest mem and starts executing on it) before another process

        sbatch --job-name $asset_name config.sh $asset_name $model_name $seed_value $dataset_name $dataset_path $fine_tune_embedding_model $optimizer $learning_rate $mini_batch_size $train_epochs $dropout $lime_no_of_samples $hidden_units_classifier $hidden_units_mask_embedder $generate_explanation_for_one_instance

        config_count=$(( config_count+1 ))
    done
done

# asset_name="birnn_rnn_mask_rnn_contrast_model-MASK_OUTPUT_CORRECTED-TRAINED_ON_SERVER"
# model_name="birnn_rnn_mask_rnn_contrast"
# seed_value=11
# dataset_name="Covid-19_tweets"
# fine_tune_embedding_model=False
# optimizer="adam"
# learning_rate=3e-5
# mini_batch_size=50
# train_epochs=200
# dropout=0.4
# lime_no_of_samples=1000
# hidden_units_classifier=128
# hidden_units_mask_embedder=128
# hidden_units_contrast_embedder=128

# cpu=0
# # multiple=`expr $config_count / ${#cuda_devices[@]}`
# cuda_device_no=0

# # sshpass -p "#Deakin2630" rsync -avr --relative mask_model/ --exclude assets/ --exclude datasets/SST2_sentences/ guptashas@gandalf-dev.it.deakin.edu.au:/home/guptashas/PhD_experiments/CDR-mechanism/
# screen_name=$asset_name
# # ssh guptashas@luthin.it.deakin.edu.au "screen -d -m -S rnn_rnn_mask_rnn_contrast_model-MASK_OUTPUT_CORRECTED-TRAINED_ON_SERVER taskset --cpu-list $cpu bash mask_contrast_model/run.sh $cuda_device_no $asset_name $model_name $seed_value $dataset_name $fine_tune_embedding_model $optimizer $learning_rate $mini_batch_size $train_epochs $dropout $lime_no_of_samples $hidden_units_classifier $hidden_units_mask_embedder $hidden_units_contrast_embedder"

# screen -S ${screen_name} -d -m taskset --cpu-list $cpu bash config.sh $cuda_device_no $asset_name $model_name $seed_value $dataset_name $fine_tune_embedding_model $optimizer $learning_rate $mini_batch_size $train_epochs $dropout $lime_no_of_samples $hidden_units_classifier $hidden_units_mask_embedder $hidden_units_contrast_embedder
# # bash config.sh $cuda_device_no $asset_name $model_name $seed_value $dataset_name $fine_tune_embedding_model $optimizer $learning_rate $mini_batch_size $train_epochs $dropout $lime_no_of_samples $hidden_units_classifier $hidden_units_mask_embedder $hidden_units_contrast_embedder

# # ssh guptashas@luthin.it.deakin.edu.au "screen -d -m -L -Logfile ServerLog/`date '+%Y-%m-%d_%H:%M:%S'.txt` -S rnn_rnn_mask_rnn_contrast_model-MASK_OUTPUT_CORRECTED-TRAINED_ON_SERVER"
