import os
import pickle
import pandas as pd
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
import numpy as np
import shap
import timeit
from tqdm import tqdm

class Shap_explanations(object):
    def __init__(self, config, model, word_index):
        self.config = config
        self.model = model
        self.word_index = word_index
        self.vocab = [key for key in self.word_index.keys()]
        self.vectorize = tf.keras.layers.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)

    def prediction(self, x):
        return self.model(x, training=False)

    def create_shap_explanations(self, train_dataset):

        explanations = {"text":[], "base_value":[], "SHAP_explanation":[], "probability_output":[]}

        with open("assets/results/"+self.config["asset_name"]+".pickle", 'rb') as handle:
            results = pickle.load(handle)
        results = pd.DataFrame(results)
        train_dataset = pd.DataFrame(train_dataset)

        train_sentences = list(train_dataset.loc[(train_dataset["rule_label"]!=0)&(train_dataset["contrast"]==1)]['sentence']) + list(train_dataset.loc[(train_dataset["rule_label"]!=0)&(train_dataset["contrast"]==0)]['sentence'])
        test_sentences = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentence']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentence'])
        probabilities = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_probability_output']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_probability_output'])
        
        train_tokenized_sentences = self.vectorize(np.array(train_sentences)).numpy()
        test_tokenized_sentences = self.vectorize(np.array(test_sentences)).numpy()

        start = timeit.default_timer()
        exp_explainer = shap.Explainer(self.prediction, test_tokenized_sentences)
        exp_shap_values = exp_explainer(test_tokenized_sentences[:10])
        stop = timeit.default_timer()

        # i=0
        # print("\n")
        # print("Sentence: ", test_sentences[i])
        # print("\n")
        # print("Padded and tokenized: ", test_tokenized_sentences[i])
        # print("\n")
        # print("Model prediction: ", probabilities[i])
        # print("\n")
        # print("Base value: ", exp_shap_values.base_values[i])
        # print("\n")
        # print("Shap values: ", exp_shap_values.values[i])
        # print("\n")
        # print("Sum of Base value + Shap values: ", np.sum(exp_shap_values.values[i]) + exp_shap_values.base_values[i][0])

        explanations["text"] = test_sentences
        explanations["base_value"] = exp_shap_values.base_values
        explanations["SHAP_explanation"] = exp_shap_values.values
        explanations["probability_output"] = probabilities

        # Save the shap values
        if not os.path.exists("assets/explanations/shap_explanations"):
            os.makedirs("assets/explanations/shap_explanations")
        with open("assets/explanations/shap_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(explanations, handle)
