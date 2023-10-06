from functools import update_wrapper
from scipy.stats import ttest_ind
import math
import numpy as np
import pandas as pd
import random

def zero_scores_for_punctuations(tokens, scores):
    for index, token in enumerate(tokens):
        if token == "," or token == '`' or token == "'":
            try:
                scores[index] = 0
            except:
                continue
    return scores

def calculate_percy(results_one_rule, results_explanations, type_of_percy, K=5):
    EA_values = {
                "one_rule":[],
                "one_rule_contrast":[],
                "one_rule_no_contrast":[]
                }
    counter = 0
    results_one_rule = pd.DataFrame(results_one_rule)
    results_explanations = pd.DataFrame(results_explanations)
    
    sentences = list(results_one_rule['sentence'])
    sent_predictions = list(results_one_rule['sentiment_prediction_output'])
    sent_labels = list(results_one_rule['sentiment_label'])
    rule_labels = list(results_one_rule['rule_label'])
    contrasts = list(results_one_rule['contrast'])
    # sentences = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentence']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentence'])
    # rule_labels = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['rule_label']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['rule_label'])
    # contrasts = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['contrast']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['contrast'])
    # sent_predictions = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_prediction_output']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_prediction_output'])
    # sent_labels = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_label']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_label'])
    try:
        features = list(results_explanations["features"])
    except:
        features = list(results_explanations["sentence"])
    explanations = list(results_explanations["LIME_explanation_normalised"])

    # Anecdotal examples
    corr_sent_pred_wrong_percy_score = {"sentence":[],
                                        "sentiment":[],
                                        "a_conjunct":[],
                                        "b_conjunct":[],
                                        "a_conjunct_explanations":[],
                                        "b_conjunct_explanations":[],
                                        "a_conjunct_score":[],
                                        "b_conjunct_score":[]}

    for index, sentence in enumerate(sentences):
        
        # Select LIME explanations corresponding to those tokens
        exp = explanations[index]
        
        # Check 1: Drop the sentences for which LIME explanation couldn't be calculated
        if explanations[index] == "couldn't process":
            continue

        # Check 2: If A&B conjuncts contains 1 token atleast
        tokenized_sentence = sentence.split()
        if rule_labels[index] == 1:
            rule_word = "but"
        elif rule_labels[index] == 2:
            rule_word = "yet"
        elif rule_labels[index] == 3:
            rule_word = "though"
        elif rule_labels[index] == 4:
            rule_word = "while"
        rule_word_index = tokenized_sentence.index(rule_word)
        A_conjunct = tokenized_sentence[:rule_word_index]
        B_conjunct = tokenized_sentence[rule_word_index+1:len(tokenized_sentence)]
        A_conjunct_exp = exp[0:rule_word_index]
        B_conjunct_exp = exp[rule_word_index+1:len(tokenized_sentence)]
        if len(A_conjunct) == 0 or len(B_conjunct) == 0 or len(A_conjunct_exp) == 0 or len(B_conjunct_exp)==0:
            continue
        
        # Select the tokens in Conjunct for P-value test (Fixing the number of tokens)
        A_conjunct_selected = []
        B_conjunct_selected = []
        A_conjunct_exp_sorted = sorted(A_conjunct_exp, reverse=True) # sorting tokens in descending order
        B_conjunct_exp_sorted = sorted(B_conjunct_exp, reverse=True)
        A_conjunct_exp_tokens = A_conjunct_exp_sorted[0:len(A_conjunct_exp_sorted)]
        B_conjunct_exp_tokens = B_conjunct_exp_sorted[0:len(B_conjunct_exp_sorted)]
        for value_index, value in enumerate(A_conjunct_exp_tokens):
            A_conjunct_selected.append(A_conjunct[A_conjunct_exp.index(value)])
        for value_index, value in enumerate(B_conjunct_exp_tokens):
            B_conjunct_selected.append(B_conjunct[B_conjunct_exp.index(value)])
        p_value = ttest_ind(A_conjunct_exp_tokens, B_conjunct_exp_tokens)[1] # Pvalue test to reject the null hypothesis (How does it apply in LIME-scores?)

        # Calculating the PERCY score
        if type_of_percy == "without_pval":
            if rule_word == "but" or rule_word == "yet":
                if sent_predictions[index] == sent_labels[index] and np.mean(A_conjunct_exp_tokens) < np.mean(B_conjunct_exp_tokens):
                    EA_value = 1
                    EA_values["one_rule"].append(EA_value)
                    EA_values["one_rule_contrast"].append(EA_value)
                    EA_values["one_rule_no_contrast"].append(EA_value)
                else:
                    EA_value = 0
                    EA_values["one_rule"].append(EA_value)
                    EA_values["one_rule_contrast"].append(EA_value)
                    EA_values["one_rule_no_contrast"].append(EA_value)
            elif rule_word == "though" or rule_word == "while":
                if sent_predictions[index] == sent_labels[index] and np.mean(A_conjunct_exp_tokens) > np.mean(B_conjunct_exp_tokens):
                    EA_value = 1
                    EA_values["one_rule"].append(EA_value)
                    EA_values["one_rule_contrast"].append(EA_value)
                    EA_values["one_rule_no_contrast"].append(EA_value)
                else:
                    EA_value = 0
                    EA_values["one_rule"].append(EA_value)
                    EA_values["one_rule_contrast"].append(EA_value)
                    EA_values["one_rule_no_contrast"].append(EA_value)
        elif type_of_percy == "with_pval":
            if rule_word == "but" or rule_word == "yet":
                if sent_predictions[index] == sent_labels[index] and np.mean(A_conjunct_exp_tokens) < np.mean(B_conjunct_exp_tokens) and p_value < 0.05:
                    EA_value = 1
                    EA_values["one_rule"].append(EA_value)
                    EA_values["one_rule_contrast"].append(EA_value)
                    EA_values["one_rule_no_contrast"].append(EA_value)
                else:
                    EA_value = 0
                    EA_values["one_rule"].append(EA_value)
                    EA_values["one_rule_contrast"].append(EA_value)
                    EA_values["one_rule_no_contrast"].append(EA_value)
            elif rule_word == "though" or rule_word == "while":
                if sent_predictions[index] == sent_labels[index] and np.mean(A_conjunct_exp_tokens) > np.mean(B_conjunct_exp_tokens) and p_value < 0.05:
                    EA_value = 1
                    EA_values["one_rule"].append(EA_value)
                    EA_values["one_rule_contrast"].append(EA_value)
                    EA_values["one_rule_no_contrast"].append(EA_value)
                else:
                    EA_value = 0
                    EA_values["one_rule"].append(EA_value)
                    EA_values["one_rule_contrast"].append(EA_value)
                    EA_values["one_rule_no_contrast"].append(EA_value)
        elif type_of_percy == "different_condition_for_no_contrast":
            if contrasts[index] == 1: # For rule-contrast subset
                if rule_word == "but" or rule_word == "yet":
                    if sent_predictions[index] == sent_labels[index] and np.mean(A_conjunct_exp_tokens) < np.mean(B_conjunct_exp_tokens) and p_value < 0.05:
                        EA_value = 1
                        EA_values["one_rule"].append(EA_value)
                        EA_values["one_rule_contrast"].append(EA_value)
                    else:
                        EA_value = 0
                        EA_values["one_rule"].append(EA_value)
                        EA_values["one_rule_contrast"].append(EA_value)
                elif rule_word == "though" or rule_word == "while":
                    if sent_predictions[index] == sent_labels[index] and np.mean(A_conjunct_exp_tokens) > np.mean(B_conjunct_exp_tokens) and p_value < 0.05:
                        EA_value = 1
                        EA_values["one_rule"].append(EA_value)
                        EA_values["one_rule_contrast"].append(EA_value)
                    else:
                        EA_value = 0
                        EA_values["one_rule"].append(EA_value)
                        EA_values["one_rule_contrast"].append(EA_value)
            elif contrasts[index] == 0: # For rule-no contrast subset
                if sent_predictions[index] == sent_labels[index] and p_value > 0.05:
                    EA_value = 1
                    EA_values["one_rule"].append(EA_value)
                    EA_values["one_rule_no_contrast"].append(EA_value)
                else:
                    EA_value = 0
                    EA_values["one_rule"].append(EA_value)
                    EA_values["one_rule_no_contrast"].append(EA_value)
        
        # # Was calculated during the SIGIR period (rule consistency condition is same for both contrast and no-contrast sentences)
        # elif type_of_percy == "old":
        #     if rule_word == "but" or rule_word == "yet":
        #         if sent_predictions[index] == sent_labels[index] and sum(A_conjunct_exp_tokens) < sum(B_conjunct_exp_tokens):
        #             EA_value = 1
        #             EA_values["one_rule"].append(EA_value)
        #             EA_values["one_rule_contrast"].append(EA_value)
        #             EA_values["one_rule_no_contrast"].append(EA_value)
        #         else:
        #             EA_value = 0
        #             EA_values["one_rule"].append(EA_value)
        #             EA_values["one_rule_contrast"].append(EA_value)
        #             EA_values["one_rule_no_contrast"].append(EA_value)
        #     elif rule_word == "though" or rule_word == "while":
        #         if sent_predictions[index] == sent_labels[index] and sum(A_conjunct_exp_tokens) > sum(B_conjunct_exp_tokens):
        #             EA_value = 1
        #             EA_values["one_rule"].append(EA_value)
        #             EA_values["one_rule_contrast"].append(EA_value)
        #             EA_values["one_rule_no_contrast"].append(EA_value)
        #         else:
        #             EA_value = 0
        #             EA_values["one_rule"].append(EA_value)
        #             EA_values["one_rule_contrast"].append(EA_value)
        #             EA_values["one_rule_no_contrast"].append(EA_value)
    
    return EA_values

def calculate_percy_cikm_2021(results_one_rule, results_explanations, K=10):
    EA_values = {
                "one_rule":[],
                "one_rule_contrast":[],
                "one_rule_no_contrast":[]
                }
    counter = 0
    results_one_rule = pd.DataFrame(results_one_rule)
    results_explanations = pd.DataFrame(results_explanations)
    
    sentences = list(results_one_rule['sentence'])
    sent_predictions = list(results_one_rule['sentiment_prediction_output'])
    sent_labels = list(results_one_rule['sentiment_label'])
    rule_labels = list(results_one_rule['rule_label'])
    contrasts = list(results_one_rule['contrast'])
    explanations = list(results_explanations["LIME_explanation_normalised"])

    for index, sentence in enumerate(sentences):
        
        # Select LIME explanations corresponding to those tokens
        explanation = explanations[index]
        
        # Check 1: Drop the sentences for which LIME explanation couldn't be calculated
        if explanation == "couldn't process":
            continue

        # Check 2: If A&B conjuncts contains 1 token atleast
        tokenized_sentence = sentence.split()
        if rule_labels[index] == 1:
            rule_word = "but"
        elif rule_labels[index] == 2:
            rule_word = "yet"
        elif rule_labels[index] == 3:
            rule_word = "though"
        elif rule_labels[index] == 4:
            rule_word = "while"
        rule_word_index = tokenized_sentence.index(rule_word)

        if rule_word_index - K < 0 or rule_word_index - K > len(tokenized_sentence):
            continue
        
        # Select the tokens in Conjunct for P-value test (Fixing the number of tokens)
        window = explanation[rule_word_index - K:rule_word_index + K +1]
        mid = int(len(window)/2)
        A_conjunct_exp = window[0:mid]
        B_conjunct_exp = window[mid+1:len(window)]
        p_value = ttest_ind(A_conjunct_exp, B_conjunct_exp)[1] # Pvalue test to reject the null hypothesis (How does it apply in LIME-scores?)

        # Calculating the PERCY score
        if rule_word == "but" or rule_word == "yet":
            if sent_predictions[index] == sent_labels[index] and np.max(A_conjunct_exp) < np.max(B_conjunct_exp) and p_value < 0.05:
                EA_value = 1
                EA_values["one_rule"].append(EA_value)
                EA_values["one_rule_contrast"].append(EA_value)
                EA_values["one_rule_no_contrast"].append(EA_value)
            else:
                EA_value = 0
                EA_values["one_rule"].append(EA_value)
                EA_values["one_rule_contrast"].append(EA_value)
                EA_values["one_rule_no_contrast"].append(EA_value)
        elif rule_word == "though" or rule_word == "while":
            if sent_predictions[index] == sent_labels[index] and np.max(A_conjunct_exp) > np.max(B_conjunct_exp) and p_value < 0.05:
                EA_value = 1
                EA_values["one_rule"].append(EA_value)
                EA_values["one_rule_contrast"].append(EA_value)
                EA_values["one_rule_no_contrast"].append(EA_value)
            else:
                EA_value = 0
                EA_values["one_rule"].append(EA_value)
                EA_values["one_rule_contrast"].append(EA_value)
                EA_values["one_rule_no_contrast"].append(EA_value)
    
    return EA_values