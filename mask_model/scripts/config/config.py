import os
import pickle

ASSET_NAME = "lstm_bilstm_mask_model-128_hidden_units-trained_on_PC" # change manually
MODEL_NAME = "lstm_bilstm_mask"
SEED_VALUE = 11
DATASET_NAME = "Covid-19_tweets"
FINE_TUNE_EMBEDDING_MODEL = False # False => "static", True => "non_static"
OPTIMIZER = "adam" # adam, adadelta (change manually)
LEARNING_RATE = 3e-5 # 1e-5, 3e-5, 5e-5, 10e-5
MINI_BATCH_SIZE = 50 # 30, 50
TRAIN_EPOCHS = 200
DROPOUT = 0.4
LIME_NO_OF_SAMPLES = 1000
HIDDEN_UNITS_CLASSIFIER = 128
HIDDEN_UNITS_MASK_EMBEDDER = 128

def load_configuration_parameters():
    config = {"asset_name":ASSET_NAME,
                "model_name":MODEL_NAME,
                "seed_value":SEED_VALUE,
                "dataset_name":DATASET_NAME,
                "fine_tune_embedding_model":FINE_TUNE_EMBEDDING_MODEL,
                "optimizer":OPTIMIZER,
                "learning_rate":LEARNING_RATE, 
                "mini_batch_size":MINI_BATCH_SIZE, 
                "train_epochs":TRAIN_EPOCHS,
                "dropout":DROPOUT,
                "lime_no_of_samples":LIME_NO_OF_SAMPLES,
                "hidden_units_classifier":HIDDEN_UNITS_CLASSIFIER,
                "hidden_units_mask_embedder":HIDDEN_UNITS_MASK_EMBEDDER,}
    return config

# mask_models = []
# masks = ["rnn", "birnn", "gru", "bigru", "lstm", "bilstm"]    
# for mask in masks:
#     # models = ["rnn_"+mask+"_mask_model-128-testing_reproducibility", 
#     #           "birnn_"+mask+"_mask_model-128-testing_reproducibility", 
#     #           "gru_"+mask+"_mask_model-128-testing_reproducibility", 
#     #           "bigru_"+mask+"_mask_model-128-testing_reproducibility", 
#     #           "lstm_"+mask+"_mask_model-128-testing_reproducibility", 
#     #           "bilstm_"+mask+"_mask_model-128-testing_reproducibility"]
#     models = ["rnn_"+mask+"_mask_model", 
#               "birnn_"+mask+"_mask_model", 
#               "gru_"+mask+"_mask_model", 
#               "bigru_"+mask+"_mask_model", 
#               "lstm_"+mask+"_mask_model", 
#               "bilstm_"+mask+"_mask_model"]
#     mask_models.append(models)
# for index_1, mask_model in enumerate(mask_models):
#     for index_2, model in enumerate(mask_model):
#         if model == "birnn_rnn_mask_model":
#             mask_models[index_1][index_2] = "birnn_rnn_mask_model-256_HIDDEN_UNITS_CLASSIFIER"
#         if model == "birnn_birnn_mask_model":
#             mask_models[index_1][index_2] = "birnn_birnn_mask_model-256_HIDDEN_UNITS_CLASSIFIER"
#         if model == "birnn_gru_mask_model":
#             mask_models[index_1][index_2] = "birnn_gru_mask_model-256_HIDDEN_UNITS_CLASSIFIER"
#         if model == "birnn_bigru_mask_model":
#             mask_models[index_1][index_2] = "birnn_bigru_mask_model-256_HIDDEN_UNITS_CLASSIFIER"
#         if model == "birnn_lstm_mask_model":
#             mask_models[index_1][index_2] = "birnn_lstm_mask_model-256_HIDDEN_UNITS_CLASSIFIER"
#         if model == "birnn_bilstm_mask_model":
#             mask_models[index_1][index_2] = "birnn_bilstm_mask_model-256_HIDDEN_UNITS_CLASSIFIER"
#         if model == "gru_gru_mask_model":
#             mask_models[index_1][index_2] = "gru_gru_mask_model-3"
#         if model == "lstm_bigru_mask_model":
#             mask_models[index_1][index_2] = "lstm_bigru_mask_model-2"
#         if model == "bigru_gru_mask_model":
#             mask_models[index_1][index_2] = "bigru_gru_mask_model-2"
#         if model == "lstm_gru_mask_model":
#             mask_models[index_1][index_2] = "lstm_gru_mask_model-2"
#         if model == "bilstm_gru_mask_model":
#             mask_models[index_1][index_2] = "bilstm_gru_mask_model-2"
# #         if model == "lstm_bilstm_mask_model-128-testing_reproducibility":
# #             mask_models[index_1][index_2] = "lstm_bilstm_mask_model-128_hidden_units-trained_on_PC"