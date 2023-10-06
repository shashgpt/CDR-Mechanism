import os
import pickle
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
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

    def prediction(self, x):
        # return self.model(x)
        return self.model.predict(x)
    
    def vectorize(self, texts):
        if tf.executing_eagerly():
            vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab, pad_to_max_tokens=False)
            return vectorize_layer(texts).numpy()
        else:
            toknized_texts = [text.split() for text in texts]
            encoded_texts = []
            for text in toknized_texts:
                encoded_text = []
                for word in text:
                    encoded_text.append(self.vocab.index(word)) 
                encoded_texts.append(encoded_text)
            encoded_texts = np.array(encoded_texts)
            padded_encoded_texts = tf.keras.preprocessing.sequence.pad_sequences(encoded_texts, value=0, padding='post', maxlen=71)
            return padded_encoded_texts

    def create_shap_explanations(self, train_dataset):

        explanations = {"text":[], "base_value":[], "SHAP_explanation":[], "probability_output":[]}

        with open("assets/results/"+self.config["asset_name"]+".pickle", 'rb') as handle:
            results = pickle.load(handle)
        results = pd.DataFrame(results)
        train_dataset = pd.DataFrame(train_dataset)

        train_sentences = list(train_dataset.loc[(train_dataset["rule_label"]!=0)&(train_dataset["contrast"]==1)]['sentence']) + list(train_dataset.loc[(train_dataset["rule_label"]!=0)&(train_dataset["contrast"]==0)]['sentence'])
        test_sentences = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentence']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentence'])
        probabilities = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_probability_output']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_probability_output'])
        
        # train_tokenized_sentences = self.vectorize(np.array(train_sentences)).numpy()
        # test_tokenized_sentences = self.vectorize(np.array(test_sentences)).numpy()
        train_tokenized_sentences = self.vectorize(np.array(train_sentences))
        test_tokenized_sentences = self.vectorize(np.array(test_sentences))
        print(train_tokenized_sentences.shape)
        print(test_tokenized_sentences.shape)

        start = timeit.default_timer()
        exp_explainer = shap.Explainer(model=self.prediction, masker=test_sentences[:100], algorithm="auto")
        shap_explanations = exp_explainer(test_sentences[:10])
        # exp_explainer = shap.DeepExplainer(self.model, test_sentences[:100])
        # shap_values = exp_explainer.shap_values(test_sentences[:10])
        stop = timeit.default_timer()

        # for i in range(3):
        #     print("\n")
        #     print("Sentence: ", test_sentences[i])
        #     print("\n")
        #     print("Padded and tokenized: ", test_tokenized_sentences[i])
        #     print("\n")
        #     print("Len of tokenized sentence: ", len([x for x in test_tokenized_sentences[i] if x > 0]))
        #     print("\n")
        #     print("Model prediction: ", probabilities[i])
        #     print("\n")
        #     print("Base value: ", exp_shap_values.base_values[i])
        #     print("\n")
        #     print("Shap values: ", exp_shap_values.values[i])
        #     print("\n")
        #     print("No of non-zero SHAP values: ", np.count_nonzero(exp_shap_values.values[i]))
        #     print("Sum of Base value + Shap values: ", np.sum(exp_shap_values.values[i]) + exp_shap_values.base_values[i][0])
        

        # explanations["text"] = test_sentences
        # explanations["base_value"] = exp_shap_values.base_values
        # explanations["SHAP_explanation"] = exp_shap_values.values
        # explanations["probability_output"] = probabilities

        # # Save the shap values
        # if not os.path.exists("assets/explanations/shap_explanations"):
        #     os.makedirs("assets/explanations/shap_explanations")
        # with open("assets/explanations/shap_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
        #     pickle.dump(explanations, handle)
