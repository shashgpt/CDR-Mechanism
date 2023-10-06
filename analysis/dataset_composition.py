import pickle
import numpy as np
import pandas as pd


class Dataset_composition(object):

    def create_dataset_compositions(self, dataset_name):

        with open("base_model/assets/input_dataset/"+dataset_name+"/"+"dataset.pickle", "rb") as handle:
            dataset = pickle.load(handle)
        with open("base_model/assets/input_dataset/"+dataset_name+"/"+"train_dataset.pickle", "rb") as handle:
            train_dataset = pickle.load(handle)
        with open("base_model/assets/input_dataset/"+dataset_name+"/"+"val_dataset.pickle", "rb") as handle:
            val_dataset = pickle.load(handle)
        with open("base_model/assets/input_dataset/"+dataset_name+"/"+"test_dataset.pickle", "rb") as handle:
            test_dataset = pickle.load(handle)

        df_dataset = pd.DataFrame(dataset)
        df_train_dataset = pd.DataFrame(train_dataset)
        df_val_dataset = pd.DataFrame(val_dataset)
        df_test_dataset = pd.DataFrame(test_dataset)

        data = {"no_rule_pos":0,
                "no_rule_neg":0,
                
                "one_rule_pos":0,
                "one_rule_neg":0,
                
                "one_rule_pos_contrast":0,
                "one_rule_neg_contrast":0,
                "one_rule_pos_no_contrast":0,
                "one_rule_neg_no_contrast":0,
                
                "a_but_b_rule_contrast_positive":0,
                "a_but_b_rule_contrast_negative":0,
                "a_but_b_rule_no_contrast_positive":0,
                "a_but_b_rule_no_contrast_negative":0,

                "a_yet_b_rule_contrast_positive":0,
                "a_yet_b_rule_contrast_negative":0,
                "a_yet_b_rule_no_contrast_positive":0,
                "a_yet_b_rule_no_contrast_negative":0,

                "a_though_b_rule_contrast_positive":0,
                "a_though_b_rule_contrast_negative":0,
                "a_though_b_rule_no_contrast_positive":0,
                "a_though_b_rule_no_contrast_negative":0,

                "a_while_b_rule_contrast_positive":0,
                "a_while_b_rule_contrast_negative":0,
                "a_while_b_rule_no_contrast_positive":0,
                "a_while_b_rule_no_contrast_negative":0
                }

        train = {"no_rule_pos":0,
                "no_rule_neg":0,
                
                "one_rule_pos":0,
                "one_rule_neg":0,
                
                "one_rule_pos_contrast":0,
                "one_rule_neg_contrast":0,
                "one_rule_pos_no_contrast":0,
                "one_rule_neg_no_contrast":0,
                
                "a_but_b_rule_contrast_positive":0,
                "a_but_b_rule_contrast_negative":0,
                "a_but_b_rule_no_contrast_positive":0,
                "a_but_b_rule_no_contrast_negative":0,

                "a_yet_b_rule_contrast_positive":0,
                "a_yet_b_rule_contrast_negative":0,
                "a_yet_b_rule_no_contrast_positive":0,
                "a_yet_b_rule_no_contrast_negative":0,

                "a_though_b_rule_contrast_positive":0,
                "a_though_b_rule_contrast_negative":0,
                "a_though_b_rule_no_contrast_positive":0,
                "a_though_b_rule_no_contrast_negative":0,

                "a_while_b_rule_contrast_positive":0,
                "a_while_b_rule_contrast_negative":0,
                "a_while_b_rule_no_contrast_positive":0,
                "a_while_b_rule_no_contrast_negative":0
                }

        val = {"no_rule_pos":0,
                "no_rule_neg":0,
                
                "one_rule_pos":0,
                "one_rule_neg":0,
                
                "one_rule_pos_contrast":0,
                "one_rule_neg_contrast":0,
                "one_rule_pos_no_contrast":0,
                "one_rule_neg_no_contrast":0,
                
                "a_but_b_rule_contrast_positive":0,
                "a_but_b_rule_contrast_negative":0,
                "a_but_b_rule_no_contrast_positive":0,
                "a_but_b_rule_no_contrast_negative":0,

                "a_yet_b_rule_contrast_positive":0,
                "a_yet_b_rule_contrast_negative":0,
                "a_yet_b_rule_no_contrast_positive":0,
                "a_yet_b_rule_no_contrast_negative":0,

                "a_though_b_rule_contrast_positive":0,
                "a_though_b_rule_contrast_negative":0,
                "a_though_b_rule_no_contrast_positive":0,
                "a_though_b_rule_no_contrast_negative":0,

                "a_while_b_rule_contrast_positive":0,
                "a_while_b_rule_contrast_negative":0,
                "a_while_b_rule_no_contrast_positive":0,
                "a_while_b_rule_no_contrast_negative":0
                }

        test = {"no_rule_pos":0,
                "no_rule_neg":0,
                
                "one_rule_pos":0,
                "one_rule_neg":0,
                
                "one_rule_pos_contrast":0,
                "one_rule_neg_contrast":0,
                "one_rule_pos_no_contrast":0,
                "one_rule_neg_no_contrast":0,
                
                "a_but_b_rule_contrast_positive":0,
                "a_but_b_rule_contrast_negative":0,
                "a_but_b_rule_no_contrast_positive":0,
                "a_but_b_rule_no_contrast_negative":0,

                "a_yet_b_rule_contrast_positive":0,
                "a_yet_b_rule_contrast_negative":0,
                "a_yet_b_rule_no_contrast_positive":0,
                "a_yet_b_rule_no_contrast_negative":0,

                "a_though_b_rule_contrast_positive":0,
                "a_though_b_rule_contrast_negative":0,
                "a_though_b_rule_no_contrast_positive":0,
                "a_though_b_rule_no_contrast_negative":0,

                "a_while_b_rule_contrast_positive":0,
                "a_while_b_rule_contrast_negative":0,
                "a_while_b_rule_no_contrast_positive":0,
                "a_while_b_rule_no_contrast_negative":0
                }

        for index, sentence in enumerate(dataset["sentence"]):

            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] == 0:
                data["no_rule_pos"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] == 0:
                data["no_rule_neg"] += 1
            
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] != 0:
                data["one_rule_pos"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] != 0:
                data["one_rule_neg"] += 1
            
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] != 0 and dataset['contrast'][index] == 1:
                data["one_rule_pos_contrast"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] != 0 and dataset['contrast'][index] == 1:
                data["one_rule_neg_contrast"] += 1
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] != 0 and dataset['contrast'][index] == 0:
                data["one_rule_pos_no_contrast"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] != 0 and dataset['contrast'][index] == 0:
                data["one_rule_neg_no_contrast"] += 1
                
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] == 1 and dataset['contrast'][index] == 1:
                data["a_but_b_rule_contrast_positive"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] == 1 and dataset['contrast'][index] == 1:
                data["a_but_b_rule_contrast_negative"] += 1
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] == 1 and dataset['contrast'][index] == 0:
                data["a_but_b_rule_no_contrast_positive"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] == 1 and dataset['contrast'][index] == 0:
                data["a_but_b_rule_no_contrast_negative"] += 1
            
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] == 2 and dataset['contrast'][index] == 1:
                data["a_yet_b_rule_contrast_positive"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] == 2 and dataset['contrast'][index] == 1:
                data["a_yet_b_rule_contrast_negative"] += 1
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] == 2 and dataset['contrast'][index] == 0:
                data["a_yet_b_rule_no_contrast_positive"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] == 2 and dataset['contrast'][index] == 0:
                data["a_yet_b_rule_no_contrast_negative"] += 1
            
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] == 3 and dataset['contrast'][index] == 1:
                data["a_though_b_rule_contrast_positive"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] == 3 and dataset['contrast'][index] == 1:
                data["a_though_b_rule_contrast_negative"] += 1
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] == 3 and dataset['contrast'][index] == 0:
                data["a_though_b_rule_no_contrast_positive"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] == 3 and dataset['contrast'][index] == 0:
                data["a_though_b_rule_no_contrast_negative"] += 1
            
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] == 4 and dataset['contrast'][index] == 1:
                data["a_while_b_rule_contrast_positive"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] == 4 and dataset['contrast'][index] == 1:
                data["a_while_b_rule_contrast_negative"] += 1
            if dataset['sentiment_label'][index] == 1 and dataset['rule_label'][index] == 4 and dataset['contrast'][index] == 0:
                data["a_while_b_rule_no_contrast_positive"] += 1
            if dataset['sentiment_label'][index] == 0 and dataset['rule_label'][index] == 4 and dataset['contrast'][index] == 0:
                data["a_while_b_rule_no_contrast_negative"] += 1
            
        for index, sentence in enumerate(train_dataset["sentence"]):

            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] == 0:
                train["no_rule_pos"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] == 0:
                train["no_rule_neg"] += 1
            
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] != 0:
                train["one_rule_pos"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] != 0:
                train["one_rule_neg"] += 1
            
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] != 0 and train_dataset['contrast'][index] == 1:
                train["one_rule_pos_contrast"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] != 0 and train_dataset['contrast'][index] == 1:
                train["one_rule_neg_contrast"] += 1
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] != 0 and train_dataset['contrast'][index] == 0:
                train["one_rule_pos_no_contrast"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] != 0 and train_dataset['contrast'][index] == 0:
                train["one_rule_neg_no_contrast"] += 1
                
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] == 1 and train_dataset['contrast'][index] == 1:
                train["a_but_b_rule_contrast_positive"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] == 1 and train_dataset['contrast'][index] == 1:
                train["a_but_b_rule_contrast_negative"] += 1
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] == 1 and train_dataset['contrast'][index] == 0:
                train["a_but_b_rule_no_contrast_positive"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] == 1 and train_dataset['contrast'][index] == 0:
                train["a_but_b_rule_no_contrast_negative"] += 1
            
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] == 2 and train_dataset['contrast'][index] == 1:
                train["a_yet_b_rule_contrast_positive"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] == 2 and train_dataset['contrast'][index] == 1:
                train["a_yet_b_rule_contrast_negative"] += 1
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] == 2 and train_dataset['contrast'][index] == 0:
                train["a_yet_b_rule_no_contrast_positive"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] == 2 and train_dataset['contrast'][index] == 0:
                train["a_yet_b_rule_no_contrast_negative"] += 1
            
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] == 3 and train_dataset['contrast'][index] == 1:
                train["a_though_b_rule_contrast_positive"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] == 3 and train_dataset['contrast'][index] == 1:
                train["a_though_b_rule_contrast_negative"] += 1
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] == 3 and train_dataset['contrast'][index] == 0:
                train["a_though_b_rule_no_contrast_positive"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] == 3 and train_dataset['contrast'][index] == 0:
                train["a_though_b_rule_no_contrast_negative"] += 1
            
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] == 4 and train_dataset['contrast'][index] == 1:
                train["a_while_b_rule_contrast_positive"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] == 4 and train_dataset['contrast'][index] == 1:
                train["a_while_b_rule_contrast_negative"] += 1
            if train_dataset['sentiment_label'][index] == 1 and train_dataset['rule_label'][index] == 4 and train_dataset['contrast'][index] == 0:
                train["a_while_b_rule_no_contrast_positive"] += 1
            if train_dataset['sentiment_label'][index] == 0 and train_dataset['rule_label'][index] == 4 and train_dataset['contrast'][index] == 0:
                train["a_while_b_rule_no_contrast_negative"] += 1

        for index, sentence in enumerate(val_dataset["sentence"]):

            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] == 0:
                val["no_rule_pos"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] == 0:
                val["no_rule_neg"] += 1
            
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] != 0:
                val["one_rule_pos"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] != 0:
                val["one_rule_neg"] += 1
            
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] != 0 and val_dataset['contrast'][index] == 1:
                val["one_rule_pos_contrast"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] != 0 and val_dataset['contrast'][index] == 1:
                val["one_rule_neg_contrast"] += 1
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] != 0 and val_dataset['contrast'][index] == 0:
                val["one_rule_pos_no_contrast"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] != 0 and val_dataset['contrast'][index] == 0:
                val["one_rule_neg_no_contrast"] += 1
                
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] == 1 and val_dataset['contrast'][index] == 1:
                val["a_but_b_rule_contrast_positive"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] == 1 and val_dataset['contrast'][index] == 1:
                val["a_but_b_rule_contrast_negative"] += 1
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] == 1 and val_dataset['contrast'][index] == 0:
                val["a_but_b_rule_no_contrast_positive"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] == 1 and val_dataset['contrast'][index] == 0:
                val["a_but_b_rule_no_contrast_negative"] += 1
            
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] == 2 and val_dataset['contrast'][index] == 1:
                val["a_yet_b_rule_contrast_positive"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] == 2 and val_dataset['contrast'][index] == 1:
                val["a_yet_b_rule_contrast_negative"] += 1
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] == 2 and val_dataset['contrast'][index] == 0:
                val["a_yet_b_rule_no_contrast_positive"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] == 2 and val_dataset['contrast'][index] == 0:
                val["a_yet_b_rule_no_contrast_negative"] += 1
            
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] == 3 and val_dataset['contrast'][index] == 1:
                val["a_though_b_rule_contrast_positive"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] == 3 and val_dataset['contrast'][index] == 1:
                val["a_though_b_rule_contrast_negative"] += 1
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] == 3 and val_dataset['contrast'][index] == 0:
                val["a_though_b_rule_no_contrast_positive"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] == 3 and val_dataset['contrast'][index] == 0:
                val["a_though_b_rule_no_contrast_negative"] += 1
            
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] == 4 and val_dataset['contrast'][index] == 1:
                val["a_while_b_rule_contrast_positive"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] == 4 and val_dataset['contrast'][index] == 1:
                val["a_while_b_rule_contrast_negative"] += 1
            if val_dataset['sentiment_label'][index] == 1 and val_dataset['rule_label'][index] == 4 and val_dataset['contrast'][index] == 0:
                val["a_while_b_rule_no_contrast_positive"] += 1
            if val_dataset['sentiment_label'][index] == 0 and val_dataset['rule_label'][index] == 4 and val_dataset['contrast'][index] == 0:
                val["a_while_b_rule_no_contrast_negative"] += 1

        for index, sentence in enumerate(test_dataset["sentence"]):

            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] == 0:
                test["no_rule_pos"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] == 0:
                test["no_rule_neg"] += 1
            
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] != 0:
                test["one_rule_pos"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] != 0:
                test["one_rule_neg"] += 1
            
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] != 0 and test_dataset['contrast'][index] == 1:
                test["one_rule_pos_contrast"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] != 0 and test_dataset['contrast'][index] == 1:
                test["one_rule_neg_contrast"] += 1
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] != 0 and test_dataset['contrast'][index] == 0:
                test["one_rule_pos_no_contrast"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] != 0 and test_dataset['contrast'][index] == 0:
                test["one_rule_neg_no_contrast"] += 1
                
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] == 1 and test_dataset['contrast'][index] == 1:
                test["a_but_b_rule_contrast_positive"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] == 1 and test_dataset['contrast'][index] == 1:
                test["a_but_b_rule_contrast_negative"] += 1
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] == 1 and test_dataset['contrast'][index] == 0:
                test["a_but_b_rule_no_contrast_positive"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] == 1 and test_dataset['contrast'][index] == 0:
                test["a_but_b_rule_no_contrast_negative"] += 1
            
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] == 2 and test_dataset['contrast'][index] == 1:
                test["a_yet_b_rule_contrast_positive"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] == 2 and test_dataset['contrast'][index] == 1:
                test["a_yet_b_rule_contrast_negative"] += 1
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] == 2 and test_dataset['contrast'][index] == 0:
                test["a_yet_b_rule_no_contrast_positive"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] == 2 and test_dataset['contrast'][index] == 0:
                test["a_yet_b_rule_no_contrast_negative"] += 1
            
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] == 3 and test_dataset['contrast'][index] == 1:
                test["a_though_b_rule_contrast_positive"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] == 3 and test_dataset['contrast'][index] == 1:
                test["a_though_b_rule_contrast_negative"] += 1
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] == 3 and test_dataset['contrast'][index] == 0:
                test["a_though_b_rule_no_contrast_positive"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] == 3 and test_dataset['contrast'][index] == 0:
                test["a_though_b_rule_no_contrast_negative"] += 1
            
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] == 4 and test_dataset['contrast'][index] == 1:
                test["a_while_b_rule_contrast_positive"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] == 4 and test_dataset['contrast'][index] == 1:
                test["a_while_b_rule_contrast_negative"] += 1
            if test_dataset['sentiment_label'][index] == 1 and test_dataset['rule_label'][index] == 4 and test_dataset['contrast'][index] == 0:
                test["a_while_b_rule_no_contrast_positive"] += 1
            if test_dataset['sentiment_label'][index] == 0 and test_dataset['rule_label'][index] == 4 and test_dataset['contrast'][index] == 0:
                test["a_while_b_rule_no_contrast_negative"] += 1
        
        return data, train, val, test, df_dataset, df_train_dataset, df_val_dataset, df_test_dataset