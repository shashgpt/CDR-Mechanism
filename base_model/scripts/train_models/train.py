import os
from random import shuffle
import shutil
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

from scripts.train_models.additional_validation_sets import AdditionalValidationSets

class Train(object):
    def __init__(self, config, word_index):
        self.config = config
        self.word_index = word_index
        self.vocab = [key for key in self.word_index.keys()]
        # self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)
        self.vectorize_layer = tf.keras.layers.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)
    
    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        vectorized_sentences = self.vectorize_layer(np.array(sentences))
        return vectorized_sentences.numpy()

    def train_model(self, model, train_dataset, val_datasets, test_datasets):
        
        # Make paths
        if not os.path.exists("assets/training_log/"):
            os.makedirs("assets/training_log/")

        # Create train and val datasets
        train_sentences = self.vectorize(train_dataset["sentence"])
        train_sentiment_labels = np.array(train_dataset["sentiment_label"])

        val_sentences = self.vectorize(val_datasets["val_dataset"]["sentence"])
        val_sentiment_labels = np.array(val_datasets["val_dataset"]["sentiment_label"])

        train_dataset = (train_sentences, train_sentiment_labels) 
        val_dataset = (val_sentences, val_sentiment_labels)

        # Additional datasets for calculating metrics per epoch
        additional_validation_datasets = []
        for key, value in test_datasets.items():
            # if key in ["test_dataset_one_rule"]:
            #     continue
            sentences = self.vectorize(test_datasets[key]["sentence"])
            sentiment_labels = np.array(test_datasets[key]["sentiment_label"])
            dataset = (sentences, sentiment_labels, key)
            additional_validation_datasets.append(dataset)

        # Define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',              # 1. Calculate val_loss_1 
                                                                    min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                                    patience=10,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                    verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                                    mode="min",
                                                                    baseline=None, 
                                                                    restore_best_weights=True)
        my_callbacks = [early_stopping_callback, AdditionalValidationSets(additional_validation_datasets, self.config)]
        model.fit(x=train_dataset[0], 
                    y=train_dataset[1], 
                    epochs=self.config["train_epochs"], 
                    batch_size=self.config["mini_batch_size"], 
                    validation_data=val_dataset, 
                    callbacks=my_callbacks,
                    shuffle=False)
        
        # Save weights of the model
        if not os.path.exists("assets/trained_models/"):
            os.makedirs("assets/trained_models/")
        model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

    