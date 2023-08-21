import os
import pickle
import numpy as np
import tensorflow as tf



class Evaluation(object):
    def __init__(self, config, word_index):
        self.config = config
        self.word_index = word_index
        self.vocab = [key for key in self.word_index.keys()]
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)
    
    def vectorize(self, sentences):
        """
        tokenize each sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        return self.vectorize_layer(np.array(sentences)).numpy()
    
    def pad_rule_mask(self, rule_masks):
        """
        right pad each rule mask with 5 till max token length sentence
        """
        return tf.keras.preprocessing.sequence.pad_sequences(rule_masks, value=5, padding='post')

    def evaluate_model(self, model, test_datasets):

        # Results to be created after evaluation
        results = {'sentence':[], 
                    'sentiment_label':[],  
                    'rule_label':[],
                    'rule_label_mask':[],
                    'contrast':[],
                    'sentiment_probability_output':[], 
                    'sentiment_prediction_output':[],
                    'rule_label_mask_probability_output':[],
                    'rule_label_mask_prediction_output':[],
                    'contrast_probability_output':[],
                    'contrast_prediction_output':[]}

        # Create numpy arrays
        test_dataset = test_datasets["test_dataset"]
        test_sentences = self.vectorize(test_dataset["sentence"])
        test_sentiment_labels = np.array(test_dataset["sentiment_label"])
        test_rule_masks = self.pad_rule_mask(test_dataset["rule_mask"])
        test_rule_masks = test_rule_masks.reshape(test_rule_masks.shape[0], test_rule_masks.shape[1], 1)
        test_contrast_labels = np.array(test_dataset["contrast"])

        dataset = (test_sentences, [test_sentiment_labels, test_rule_masks, test_contrast_labels])

        # Evaluation and predictions
        evaluations = model.evaluate(x=dataset[0], y=dataset[1])
        print("test loss, test acc:", evaluations)
        predictions = model.predict(x=dataset[0])

        for index, sentence in enumerate(test_dataset["sentence"]):
            results['sentence'].append(test_dataset['sentence'][index])
            results['sentiment_label'].append(test_dataset['sentiment_label'][index])
            results['rule_label'].append(test_dataset['rule_label'][index])
            results['contrast'].append(test_dataset['contrast'][index])
            results['rule_label_mask'].append(test_dataset['rule_mask'][index])
        for index, prediction in enumerate(predictions[0]):
            results['sentiment_probability_output'].append(prediction)
            prediction = np.rint(prediction)
            results['sentiment_prediction_output'].append(prediction[0])
        for index, prediction in enumerate(predictions[1]):
            prediction = prediction[:len(test_dataset['rule_mask'][index])]
            prediction = prediction.reshape(prediction.shape[0])
            results["rule_label_mask_probability_output"].append(prediction)
            prediction = np.rint(prediction)
            results["rule_label_mask_prediction_output"].append(prediction)
        for prediction in predictions[2]:
            results["contrast_probability_output"].append(prediction)
            prediction = np.rint(prediction)
            results['contrast_prediction_output'].append(prediction[0])
        
        # Save the results
        if not os.path.exists("assets/results/"):
            os.makedirs("assets/results/")
        with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(results, handle)