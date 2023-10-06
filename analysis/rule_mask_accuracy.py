import numpy as np
import pandas as pd

def calculate_rule_mask_accuracy(results_mask):

    rule_mask_corrects = {'overall':None, 
                        'no_rule':None,
                        'one_rule':None,
                        'one_rule_contrast':None,
                        'one_rule_no_contrast':None,
                        'a_but_b':None, 
                        'a_but_b_contrast':None, 
                        'a_but_b_no_contrast':None,
                        'a_yet_b':None, 
                        'a_yet_b_contrast':None, 
                        'a_yet_b_no_contrast':None,
                        'a_though_b':None, 
                        'a_though_b_contrast':None, 
                        'a_though_b_no_contrast':None,
                        'a_while_b':None, 
                        'a_while_b_contrast':None, 
                        'a_while_b_no_contrast':None}
    
    results_mask = pd.DataFrame(results_mask)
    results_mask_one_rule_contrast = results_mask.loc[(results_mask["rule_label"]!=0)&(results_mask["contrast"]==1)].reset_index(drop=True)
    results_mask_one_rule_no_contrast = results_mask.loc[(results_mask["rule_label"]!=0)&(results_mask["contrast"]==0)].reset_index(drop=True)

    #contrast subset
    contrast_scores = []
    for index, _ in enumerate(results_mask_one_rule_contrast.iterrows()):
        if results_mask_one_rule_contrast["rule_label"][index] == 1:
            tokenized_sentence = results_mask_one_rule_contrast["sentence"][index].split()
            but_index = tokenized_sentence.index("but")
            rule_mask_pred = results_mask_one_rule_contrast["rule_label_mask_prediction_output"][index]
            rule_mask_ground_truth = results_mask_one_rule_contrast["rule_label_mask"][index]
            assert len(rule_mask_pred) == len(rule_mask_ground_truth)
            for token_index, _ in enumerate(rule_mask_pred):
                if token_index <= but_index:
                    if rule_mask_pred[token_index] == rule_mask_ground_truth[token_index]:
                        contrast_scores.append(1)
                    elif rule_mask_pred[token_index] != rule_mask_ground_truth[token_index]:
                        contrast_scores.append(-1)
                elif token_index > but_index:
                    if rule_mask_pred[token_index] == rule_mask_ground_truth[token_index]:
                        contrast_scores.append(1)
                    elif rule_mask_pred[token_index] != rule_mask_ground_truth[token_index]:
                        contrast_scores.append(-2)
        elif results_mask_one_rule_contrast["rule_label"][index] == 2:
            tokenized_sentence = results_mask_one_rule_contrast["sentence"][index].split()
            but_index = tokenized_sentence.index("yet")
            rule_mask_pred = results_mask_one_rule_contrast["rule_label_mask_prediction_output"][index]
            rule_mask_ground_truth = results_mask_one_rule_contrast["rule_label_mask"][index]
            assert len(rule_mask_pred) == len(rule_mask_ground_truth)
            for token_index, _ in enumerate(rule_mask_pred):
                if token_index <= but_index:
                    if rule_mask_pred[token_index] == rule_mask_ground_truth[token_index]:
                        contrast_scores.append(1)
                    elif rule_mask_pred[token_index] != rule_mask_ground_truth[token_index]:
                        contrast_scores.append(-1)
                elif token_index > but_index:
                    if rule_mask_pred[token_index] == rule_mask_ground_truth[token_index]:
                        contrast_scores.append(1)
                    elif rule_mask_pred[token_index] != rule_mask_ground_truth[token_index]:
                        contrast_scores.append(-2)
        elif results_mask_one_rule_contrast["rule_label"][index] == 3:
            tokenized_sentence = results_mask_one_rule_contrast["sentence"][index].split()
            but_index = tokenized_sentence.index("though")
            rule_mask_pred = results_mask_one_rule_contrast["rule_label_mask_prediction_output"][index]
            rule_mask_ground_truth = results_mask_one_rule_contrast["rule_label_mask"][index]
            assert len(rule_mask_pred) == len(rule_mask_ground_truth)
            for token_index, _ in enumerate(rule_mask_pred):
                if token_index <= but_index:
                    if rule_mask_pred[token_index] == rule_mask_ground_truth[token_index]:
                        contrast_scores.append(1)
                    elif rule_mask_pred[token_index] != rule_mask_ground_truth[token_index]:
                        contrast_scores.append(-2)
                elif token_index > but_index:
                    if rule_mask_pred[token_index] == rule_mask_ground_truth[token_index]:
                        contrast_scores.append(1)
                    elif rule_mask_pred[token_index] != rule_mask_ground_truth[token_index]:
                        contrast_scores.append(-1)
        elif results_mask_one_rule_contrast["rule_label"][index] == 4:
            tokenized_sentence = results_mask_one_rule_contrast["sentence"][index].split()
            but_index = tokenized_sentence.index("while")
            rule_mask_pred = results_mask_one_rule_contrast["rule_label_mask_prediction_output"][index]
            rule_mask_ground_truth = results_mask_one_rule_contrast["rule_label_mask"][index]
            assert len(rule_mask_pred) == len(rule_mask_ground_truth)
            for token_index, _ in enumerate(rule_mask_pred):
                if token_index <= but_index:
                    if rule_mask_pred[token_index] == rule_mask_ground_truth[token_index]:
                        contrast_scores.append(1)
                    elif rule_mask_pred[token_index] != rule_mask_ground_truth[token_index]:
                        contrast_scores.append(-2)
                elif token_index > but_index:
                    if rule_mask_pred[token_index] == rule_mask_ground_truth[token_index]:
                        contrast_scores.append(1)
                    elif rule_mask_pred[token_index] != rule_mask_ground_truth[token_index]:
                        contrast_scores.append(-1)
                        
    #no contrast subset
    no_contrast_scores = []
    for index, row in enumerate(results_mask_one_rule_no_contrast.iterrows()):
        rule_mask_pred = results_mask_one_rule_no_contrast["rule_label_mask_prediction_output"][index]
        rule_mask_ground_truth = results_mask_one_rule_no_contrast["rule_label_mask"][index]
        assert len(rule_mask_pred) == len(rule_mask_ground_truth)
        for token_index, _ in enumerate(rule_mask_pred):
            if rule_mask_pred[token_index] == rule_mask_ground_truth[token_index]:
                no_contrast_scores.append(1)
            elif rule_mask_pred[token_index] != rule_mask_ground_truth[token_index]:
                no_contrast_scores.append(-1)
    
    rule_mask_corrects['one_rule'] = contrast_scores + no_contrast_scores
    rule_mask_corrects['one_rule_contrast'] = contrast_scores
    rule_mask_corrects['one_rule_no_contrast'] = no_contrast_scores
    return rule_mask_corrects