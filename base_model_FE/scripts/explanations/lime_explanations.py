import tensorflow as tf
import timeit
import pickle
import numpy as np
import pandas as pd
from lime import lime_text
from tqdm import tqdm
import os
import sys
import traceback


class Lime_explanations(object):
    def __init__(self, config, word_index):
        self.config = config
        self.model = None
        self.word_index = word_index
        self.vocab = [key for key in self.word_index.keys()]
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)

    def prediction(self, text):
        rule_masks = []
        for sentence in text:
            tokenized_sentence = sentence.split()
            if ('but' in tokenized_sentence and tokenized_sentence.index('but') != 0 and tokenized_sentence.index('but') != -1 and tokenized_sentence.count('but') == 1):
                a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("but")]
                b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("but")+1:]
                rule_mask = [0]*len(a_part_tokenized_sentence) + [0]*len(["but"]) + [1]*len(b_part_tokenized_sentence)
                rule_masks.append(rule_mask)
            elif ('yet' in tokenized_sentence and tokenized_sentence.index('yet') != 0 and tokenized_sentence.index('yet') != -1 and tokenized_sentence.count('yet') == 1):
                a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("yet")]
                b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("yet")+1:]
                rule_mask = [0]*len(a_part_tokenized_sentence) + [0]*len(["yet"]) + [1]*len(b_part_tokenized_sentence)
                rule_masks.append(rule_mask)
            elif ('though' in tokenized_sentence and tokenized_sentence.index('though') != 0 and tokenized_sentence.index('though') != -1 and tokenized_sentence.count('though') == 1):
                a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("though")]
                b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("though")+1:]
                rule_mask = [1]*len(a_part_tokenized_sentence) + [0]*len(["though"]) + [0]*len(b_part_tokenized_sentence)
                rule_masks.append(rule_mask)
            elif ('while' in tokenized_sentence and tokenized_sentence.index('while') != 0 and tokenized_sentence.index('while') != -1 and tokenized_sentence.count('while') == 1):
                a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("while")]
                b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("while")+1:]
                rule_mask = [1]*len(a_part_tokenized_sentence) + [0]*len(["while"]) + [0]*len(b_part_tokenized_sentence)
                rule_masks.append(rule_mask)
            else:
                mask_length = len(tokenized_sentence)
                rule_mask = [1]*mask_length
                rule_masks.append(rule_mask)
        x = self.vectorize_layer(np.array(text)).numpy()
        rule_masks_padded = tf.keras.preprocessing.sequence.pad_sequences(rule_masks, value=5, padding='post')
        rule_masks_padded_reshaped = rule_masks_padded.reshape(rule_masks_padded.shape[0], rule_masks_padded.shape[1], 1)
        # assert (x.shape==rule_masks_padded.shape)
        pred_prob_1 = self.model.predict([x,rule_masks_padded_reshaped], batch_size=self.config["lime_no_of_samples"])
        pred_prob_0 = 1 - pred_prob_1
        prob = np.concatenate((pred_prob_0, pred_prob_1), axis=1)
        return prob

    def create_lime_explanations(self, model):

        model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
        self.model =  model

        explanations = {"sentence":[], "LIME_explanation":[], "LIME_explanation_normalised":[]}

        with open("assets/results/"+self.config["asset_name"]+".pickle", 'rb') as handle:
            results = pickle.load(handle)
        results = pd.DataFrame(results)

        sentences = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentence']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentence'])
        # rule_masks = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['rule_mask']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['rule_mask'])
        probabilities = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_probability_output']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_probability_output'])

        explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], split_expression=" ", random_state=self.config["seed_value"])

        for index, test_datapoint in enumerate(tqdm(sentences)):
            probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
            tokenized_sentence = test_datapoint.split()
            try:
                exp = explainer.explain_instance(test_datapoint, self.prediction, num_features = len(tokenized_sentence), num_samples=self.config["lime_no_of_samples"])
            except:
                traceback.print_exc()
                text = test_datapoint
                explanation = "couldn't process"
                explanations["sentence"].append(text)
                explanations["LIME_explanation"].append(explanation)
                explanations["LIME_explanation_normalised"].append(explanation)
                continue
            text = []
            explanation = []
            explanation_normalised = []
            for word in test_datapoint.split():
                for weight in exp.as_list():
                    weight = list(weight)
                    if weight[0]==word:
                        text.append(word)
                        if weight[1] < 0:
                            weight_normalised_negative_class = abs(weight[1])*probability[0]
                            explanation_normalised.append(weight_normalised_negative_class)
                        elif weight[1] > 0:
                            weight_normalised_positive_class = abs(weight[1])*probability[1]
                            explanation_normalised.append(weight_normalised_positive_class)
                        explanation.append(weight[1])
            explanations['sentence'].append(text)
            explanations['LIME_explanation'].append(explanation)
            explanations['LIME_explanation_normalised'].append(explanation_normalised)

            if self.config["generate_explanation_for_one_instance"] == True:
                break
        
        if not os.path.exists("assets/lime_explanations/"):
            os.makedirs("assets/lime_explanations/")
        with open("assets/lime_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(explanations, handle)