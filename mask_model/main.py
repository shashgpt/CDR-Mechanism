import logging
import sys
import os
import pickle
import numpy as np
import random
import tensorflow as tf
import warnings
import pandas as pd
import argparse
import subprocess as sp
import pprint
from tensorflow.keras.utils import plot_model

# Scrips imports
# from scripts.config.config import load_configuration_parameters
from scripts.dataset_processing.preprocess_dataset import Preprocess_dataset
from scripts.dataset_processing.word_vectors import Word_vectors
from scripts.dataset_processing.dataset_division import Dataset_division
from scripts.models.models import *
from scripts.train_models.train import Train
from scripts.evaluate_models.evaluation import Evaluation
# from scripts.explanations.shap_explanations import Shap_explanations
from scripts.explanations.lime_explanations import Lime_explanations

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Change the code execution directory to current directory
os.chdir(os.getcwd())

# Disable warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# set the gpu execution device with highest free memory
def mask_unused_gpus(leave_unmasked=1): # No of avaialbe GPUs on the system
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        available_gpus = [i for i, x in enumerate(memory_free_values)]
        if len(available_gpus) < leave_unmasked: raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
        gpu_with_highest_free_memory = 0
        highest_free_memory = 0
        for index, memory in enumerate(memory_free_values):
            if memory > highest_free_memory:
                highest_free_memory = memory
                gpu_with_highest_free_memory = index
        return str(gpu_with_highest_free_memory)
    except Exception as e:
        print('\n"nvidia-smi" is probably not installed. GPUs are not masked', e)
os.environ["CUDA_VISIBLE_DEVICES"] = mask_unused_gpus()

# set the memory growth to true
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__=='__main__':

    # Gather configuration parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_name",
                        type=str,
                        required=True)
    parser.add_argument("--model_name",
                        type=str,
                        required=True)
    parser.add_argument("--seed_value",
                        type=int,
                        required=True)
    parser.add_argument("--dataset_name",
                        type=str,
                        required=True)
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True)
    parser.add_argument("--fine_tune_embedding_model",
                        type=bool,
                        required=True)  
    parser.add_argument("--optimizer",
                        type=bool,
                        required=True)
    parser.add_argument("--learning_rate",
                        type=float,
                        required=True)
    parser.add_argument("--mini_batch_size",
                        type=int,
                        required=True)
    parser.add_argument("--train_epochs",
                        type=int,
                        required=True)
    parser.add_argument("--dropout",
                        type=float,
                        required=True)
    parser.add_argument("--lime_no_of_samples",
                        type=int,
                        required=True)
    parser.add_argument("--hidden_units_classifier",
                        type=int,
                        required=True)
    parser.add_argument("--hidden_units_mask_embedder",
                        type=int,
                        required=True)
    parser.add_argument("--generate_explanation_for_one_instance",
                        type=bool,
                        required=True)
    args = parser.parse_args()
    config = vars(args)
    print("\n")
    pprint.pprint(config)
    print("\n")

    #log the output
    old_stdout = sys.stdout
    if not os.path.exists("assets/run_logs/"):
        os.makedirs("assets/run_logs/")
    log_file = open("assets/run_logs/"+config["asset_name"]+".txt","w")
    sys.stdout = log_file

    # Set seed value
    os.environ['PYTHONHASHSEED']=str(config["seed_value"])
    random.seed(config["seed_value"])
    np.random.seed(config["seed_value"])
    tf.random.set_seed(config["seed_value"])

    # # Create input data for model
    # print("\nCreating input data")
    # raw_dataset = pickle.load(open("datasets/"+config["dataset_name"]+"/raw_dataset.pickle", "rb"))
    # raw_dataset = pd.DataFrame(raw_dataset)
    # preprocessed_dataset = Preprocess_dataset(config).preprocess_covid_tweets(raw_dataset)
    # preprocessed_dataset = pd.DataFrame(preprocessed_dataset)
    # word_vectors, word_index = Word_vectors(config).create_word_vectors(preprocessed_dataset)
    # train_dataset, val_datasets, test_datasets = Dataset_division(config).train_val_test_split(preprocessed_dataset)
    
    # Reading existing created datasets and word vectors
    # preprocessed_dataset = pickle.load(open("datasets/"+config["dataset_name"]+"/preprocessed_dataset.pickle", "rb"))
    # preprocessed_dataset = pd.DataFrame(preprocessed_dataset)
    with open("datasets/"+config["dataset_name"]+"/"+"/word_index.pickle", "rb") as handle:
        word_index = pickle.load(handle)
    with open("datasets/"+config["dataset_name"]+"/"+"/word_vectors.npy", "rb") as handle:
        word_vectors = np.load(handle)
    train_dataset = pickle.load(open("datasets/"+config["dataset_name"]+"/train_dataset.pickle", "rb"))
    val_datasets = pickle.load(open("datasets/"+config["dataset_name"]+"/val_dataset.pickle", "rb"))
    test_datasets = pickle.load(open("datasets/"+config["dataset_name"]+"/test_dataset.pickle", "rb"))

    # Create model
    print("\nBuilding model")
    if config["model_name"] == "rnn_bilstm_mask":
        model = rnn_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_bilstm_mask":
        model = birnn_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_bilstm_mask":
        model = gru_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_bilstm_mask":
        model = bigru_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_bilstm_mask":
        model = lstm_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_bilstm_mask":
        model = bilstm_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)

    elif config["model_name"] == "rnn_bigru_mask":
        model = rnn_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_bigru_mask":
        model = birnn_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_bigru_mask":
        model = gru_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_bigru_mask":
        model = bigru_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_bigru_mask":
        model = lstm_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_bigru_mask":
        model = bilstm_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)

    elif config["model_name"] == "rnn_lstm_mask":
        model = rnn_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_lstm_mask":
        model = birnn_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_lstm_mask":
        model = gru_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_lstm_mask":
        model = bigru_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_lstm_mask":
        model = lstm_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_lstm_mask":
        model = bilstm_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)

    elif config["model_name"] == "rnn_rnn_mask":
        model = rnn_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_rnn_mask":
        model = birnn_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_rnn_mask":
        model = gru_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_rnn_mask":
        model = bigru_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_rnn_mask":
        model = lstm_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_rnn_mask":
        model = bilstm_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    
    elif config["model_name"] == "rnn_gru_mask":
        model = rnn_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_gru_mask":
        model = birnn_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_gru_mask":
        model = gru_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_gru_mask":
        model = bigru_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_gru_mask":
        model = lstm_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_gru_mask":
        model = bilstm_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    
    elif config["model_name"] == "rnn_birnn_mask":
        model = rnn_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_birnn_mask":
        model = birnn_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_birnn_mask":
        model = gru_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_birnn_mask":
        model = bigru_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_birnn_mask":
        model = lstm_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_birnn_mask":
        model = bilstm_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    
    model.summary(line_length=150)
    if not os.path.exists("assets/computation_graphs"):
        os.makedirs("assets/computation_graphs")
    # plot_model(model, show_shapes=True, to_file="assets/computation_graphs/"+config["asset_name"]+".png")

    # Train model
    print("\nGPU execution device: ", mask_unused_gpus())
    print("\nTraining")
    Train(config, word_index).train_model(model, train_dataset, val_datasets, test_datasets)

    # Load trained model
    model.load_weights("assets/trained_models/"+config["asset_name"]+".h5")

    # Test model
    print("\nEvaluation")
    Evaluation(config, word_index).evaluate_model(model, test_datasets)

    # LIME explanations
    print("\nLIME explanations")
    Lime_explanations(config, model, word_index).create_lime_explanations()

    # Save the configuration parameters for this run (marks the creation of an asset)
    if not os.path.exists("assets/configurations/"):
        os.makedirs("assets/configurations/")
    with open("assets/configurations/"+config["asset_name"]+".pickle", 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #close the logfile
    sys.stdout = old_stdout
    log_file.close()

    # # Load SST2 dataset
    # x = pickle.load(open("datasets/SST2_sentences/stsa.binary.p","rb"))
    # revs, word_vectors, random_word_vectors, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]