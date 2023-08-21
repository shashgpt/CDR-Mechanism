#!/bin/bash

##### GITHUB commands #####
# # Push your project to GitHub
#   git init
#   git rm --cached . -r
#   git add --all -- ':!analysis_one_rule.ipynb' ':!analysis_one_rule_contrast.ipynb' ':!analysis_one_rule_no_contrast.ipynb' ':!commands.sh' ':!mask_model/scripts/' ':!mask_model/main.py' ':!mask_contrast_model/scripts/' ':!mask_contrast_model/main.py'
#   git status
#   git commit -m "Commit message"
#   git branch -M main
#   git remote add origin "https://github.com/shashgpt/CDR-Mechanism.git"
#   git remote -v
#   git push origin main
#   git log --oneline # get the commit ID
# # Download project at a specific commit
#   git fetch origin
#   git checkout COMMIT_ID
#   git stash or commit
# TOKEN: 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
###########################

##### SCREEN commands #####
# Kill all screens on a GCP vm: pkill screen
# Kill screens for a particular model on a GCP vm: screen -ls | egrep "lstm_rnn_mask_rnn_contrast_model-MASK_OUTPUT_CORRECTED-TRAINED_ON_SERVER" | awk -F "." '{print $1}'| xargs kill
# Start a screen: screen -S "screen" -d -m taskset --cpu-list 0 python main.py
###########################

##### DOCKER commands #####
# 0)Create an empty Dockerfile: touch Dockerfile
# 1)Pull an image from docker-hub: sudo docker pull tensorflow/tensorflow:latest-gpu-jupyter
# 2)Run the docker image:
    # sudo docker run --mount type=bind,source="$(pwd)",target=/LRD-BNN-mask --gpus all -it -p 8000:8000 --rm tensorflow/tensorflow:latest-gpu-jupyter bash
    # sudo docker run --mount type=bind,source="$(pwd)",target=/LRD-BNN-mask --gpus all -it -p 8000:8000 --rm  pytorch/pytorch:latest bash
# 3)Install libraries and develop
# 4)Commit a docker container (with all installed libraries)
    # a) Identify the container ID outside the container: sudo docker ps
    # b) Commit changes to the Docker outside the container: sudo docker commit 7dbc034efe90 tensorflow/tensorflow:latest-gpu-jupyter, sudo docker commit 7dbc034efe90 pytorch/pytorch:latest
# 5)Push the commit on docker hub outside the container
    # a) tag the image: docker tag tensorflow/tensorflow:latest-gpu-jupyter sunbro/lrd-bnn-mask/tensorflow:latest-gpu-jupyter
    # b) push the image: docker image push sunbro/lrd-bnn-mask/tensorflow:latest-gpu-jupyter
# 6)Whenever you start the container again, all the installed libraries should be loaded
# 7)Kill a docker container: sudo docker rm -f CONTAINER_NAME
###########################

##### JUPYTER NOTEBOOK commands #####
# 1)Open a Jupyter notebook inside a docker container: jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
#####################################

##### Ubuntu commands #####
# 1)Kill a process on a port on Ubuntu: sudo kill -9 `sudo lsof -t -i:8888`
###########################

##### Interacting with pippin or luthin servers #####
# sshpass -p "#Deakin2630" rsync -a --relative mask_contrast_model/ --exclude-from 'list_of_files_not_to_send_to_server.txt' guptashas@luthin.it.deakin.edu.au:/home/guptashas/PhD_experiments/CDR-mechanism/ => sending code to server
# sshpass -p "#Deakin2630" rsync -a --exclude-from 'list_of_files_not_to_receive_from_server.txt' guptashas@luthin.it.deakin.edu.au:/home/guptashas/PhD_experiments/LRD-mask/mask_model/ mask_model/ => receiving code from the server
#####################################################

##### ALWAYS Remember #####
# 1)Run the docker commit command after installing any library to save the docker container
###########################

##### Check available resources #####
# 1) GPUs: nvidia-smi
# 2) CPUs: lscpu
# 3) RAM: cat /proc/meminfo
#####################################