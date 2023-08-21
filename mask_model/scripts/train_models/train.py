import os
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
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)
        # self.vectorize_layer = tf.keras.layers.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)
    
    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        return self.vectorize_layer(np.array(sentences)).numpy()
    
    def pad_rule_mask(self, rule_masks):
        """
        right pad each rule mask with 5 till max token length sentence
        """
        return tf.keras.preprocessing.sequence.pad_sequences(rule_masks, value=5, padding='post')

    def macro_f1(self, y, y_hat, thresh=0.5):
        """Compute the macro F1-score on a batch of observations (average F1 across labels)
        
        Args:
            y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            thresh: probability value above which we predict positive
            
        Returns:
            macro_f1 (scalar Tensor): value of macro F1 for the batch
        """
        y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
        y = tf.cast(y, tf.float32)
        tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
        fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
        fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        macro_f1 = tf.reduce_mean(f1)
        return macro_f1

    def train_model(self, model, train_dataset, val_datasets, test_datasets):

        # Make paths
        if not os.path.exists("assets/training_log/"):
            os.makedirs("assets/training_log/")

        # Create Train and Val datasets
        train_sentences = self.vectorize(train_dataset["sentence"])
        train_sentiment_labels = np.array(train_dataset["sentiment_label"])
        train_rule_masks =  self.pad_rule_mask(train_dataset["rule_mask"])
        train_rule_masks = train_rule_masks.reshape(train_rule_masks.shape[0], train_rule_masks.shape[1], 1)

        val_sentences = self.vectorize(val_datasets["val_dataset"]["sentence"])
        val_sentiment_labels = np.array(val_datasets["val_dataset"]["sentiment_label"])
        val_rule_masks = self.pad_rule_mask(val_datasets["val_dataset"]["rule_mask"])
        val_rule_masks = val_rule_masks.reshape(val_rule_masks.shape[0], val_rule_masks.shape[1], 1)

        train_dataset = (train_sentences, [train_sentiment_labels, train_rule_masks])
        val_dataset = (val_sentences, [val_sentiment_labels, val_rule_masks])

        # Create additional validation datasets
        additional_validation_datasets = []
        for key, value in test_datasets.items():
            # if key in ["test_dataset_one_rule"]:
            #     continue
            sentences = self.vectorize(test_datasets[key]["sentence"])
            sentiment_labels = np.array(test_datasets[key]["sentiment_label"])
            rule_masks = self.pad_rule_mask(test_datasets[key]["rule_mask"])
            rule_masks = rule_masks.reshape(rule_masks.shape[0], rule_masks.shape[1], 1)
            dataset = (sentences, [sentiment_labels, rule_masks], key)
            additional_validation_datasets.append(dataset)

        # Define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # 1. Calculate val_loss_1 
                                                        min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                        patience=10,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                        verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                        mode="min",
                                                        baseline=None, 
                                                        restore_best_weights=True)
        my_callbacks = [early_stopping_callback, AdditionalValidationSets(additional_validation_datasets, self.config)]

        # Train the model
        model.compile(tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"]), loss=['binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])
        model.fit(x=train_dataset[0], 
                        y=train_dataset[1], 
                        batch_size=self.config["mini_batch_size"], 
                        epochs=self.config["train_epochs"], 
                        validation_data=val_dataset, 
                        callbacks=my_callbacks)

        # Save trained model
        if not os.path.exists("assets/trained_models/"):
            os.makedirs("assets/trained_models/")
        model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

        return model