import logging
import os
import pickle
import numpy as np
import random
import tensorflow as tf
import warnings
import pandas as pd
from tensorflow.keras.utils import plot_model
import argparse
import subprocess as sp
import distutils
import pprint
# from tensorflow.keras.utils import plot_model
# # from tokenizers import BertWordPieceTokenizer
# # from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs

# Scrips imports
from scripts.config.config import load_configuration_parameters
from scripts.dataset_processing.preprocess_dataset import Preprocess_dataset
from scripts.dataset_processing.word_vectors import Word_vectors
from scripts.dataset_processing.dataset_division import Dataset_division
from scripts.models.models import *
from scripts.train_models.train import Train
from scripts.evaluate_models.evaluation import Evaluation
# from scripts.explanations.shap_explanations import Shap_explanations
from scripts.explanations.lime_explanations import Lime_explanations

# Change the code execution directory to current directory
os.chdir(os.getcwd())

# Disable warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# set the gpu device with highest free memory
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
        print('"nvidia-smi" is probably not installed. GPUs are not masked', e)
os.environ["CUDA_VISIBLE_DEVICES"] = mask_unused_gpus()

# set memory growth to true
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
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=True) 
    parser.add_argument("--optimizer",
                        type=str,
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
    parser.add_argument("--generate_explanation_for_one_instance",
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=True)
    parser.add_argument("--train_model",
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=True)
    parser.add_argument("--evaluate_model",
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=True)
    parser.add_argument("--generate_explanation",
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=True)
    parser.add_argument("--prototyping",
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=False,
                        default=False)
    args = parser.parse_args()
    config = vars(args)
    if config["prototyping"] == True:
        config["asset_name"] = config["asset_name"]+"-PROTYPING"
    print("\n")
    pprint.pprint(config)
    print("\n")

    # Set seed value
    os.environ['PYTHONHASHSEED']=str(config["seed_value"])
    random.seed(config["seed_value"])
    np.random.seed(config["seed_value"])
    tf.random.set_seed(config["seed_value"])

    # print("\nCreating input data")
    if os.path.exists("datasets/"+config["dataset_name"]+"/preprocessed_dataset.pickle"):
        with open("datasets/"+config["dataset_name"]+"/"+"/word_index.pickle", "rb") as handle:
            word_index = pickle.load(handle)
        with open("datasets/"+config["dataset_name"]+"/"+"/word_vectors.npy", "rb") as handle:
            word_vectors = np.load(handle)
        train_dataset = pickle.load(open("datasets/"+config["dataset_name"]+"/train_dataset.pickle", "rb"))
        val_datasets = pickle.load(open("datasets/"+config["dataset_name"]+"/val_dataset.pickle", "rb"))
        test_datasets = pickle.load(open("datasets/"+config["dataset_name"]+"/test_dataset.pickle", "rb"))
    else:
        raw_dataset = pickle.load(open(config["dataset_path"], "rb"))
        raw_dataset = pd.DataFrame(raw_dataset)
        preprocessed_dataset = Preprocess_dataset(config).preprocess_covid_tweets(raw_dataset)
        preprocessed_dataset = pd.DataFrame(preprocessed_dataset)
        word_vectors, word_index = Word_vectors(config).create_word_vectors(preprocessed_dataset)
        train_dataset, val_datasets, test_datasets = Dataset_division(config).train_val_test_split(preprocessed_dataset)
   
    # Create model
    print("\nBuilding model")
    if config["model_name"] == "rnn":
        model = rnn(config, word_vectors)
    elif config["model_name"] == "birnn":
        model = birnn(config, word_vectors)
    elif config["model_name"] == "gru":
        model = gru(config, word_vectors)
    elif config["model_name"] == "bigru":
        model = bigru(config, word_vectors)
    elif config["model_name"] == "lstm":
        model = lstm(config, word_vectors)
    elif config["model_name"] == "bilstm":
        model = bilstm(config, word_vectors)
    model.summary(line_length = 150)
    if not os.path.exists("assets/computation_graphs"):
        os.makedirs("assets/computation_graphs")
    # plot_model(model, show_shapes = True, to_file = "assets/computation_graphs/"+config["asset_name"]+".png")

    # Train model
    if config["train_model"] == True:
        print("\nGPU execution device: ", mask_unused_gpus())
        print("\nTraining")
        Train(config, word_index).train_model(model, train_dataset, val_datasets, test_datasets)

    # Load trained model
    model.load_weights("assets/trained_models/"+config["asset_name"]+".h5")

    # Test model
    if config["evaluate_model"] == True:
        print("\nEvaluation")
        Evaluation(config, word_index).evaluate_model(model, test_datasets)

    # LIME explanations
    if config["generate_explanation"] == True:
        print("\nLIME explanations")
        Lime_explanations(config, model, word_index).create_lime_explanations()

    # Save the configuration parameters for this run (marks the creation of an asset)
    if not os.path.exists("assets/configurations/"):
        os.makedirs("assets/configurations/")
    with open("assets/configurations/"+config["asset_name"]+".pickle", 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Shap explanations
    # print("\nSHAP explanations")
    # Shap_explanations(config, model, word_index).create_shap_explanations(train_dataset)

    # # Save the configuration parameters for this run (marks the creation of an asset)
    # if not os.path.exists("assets/configurations/"):
    #     os.makedirs("assets/configurations/")
    # with open("assets/configurations/"+config["asset_name"]+".pickle", 'wb') as handle:
    #     pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)