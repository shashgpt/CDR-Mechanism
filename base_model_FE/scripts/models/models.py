import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

class Feature_extraction(tf.keras.layers.Layer):
    def __init__(self, name="feature_Extraction", **kwargs):
        super(Feature_extraction, self).__init__(name="multiply", **kwargs)
        self.supports_masking = True
        self.multiply_emb = layers.Multiply()

    def call(self, inputs):
        output = self.multiply_emb(inputs)
        return output

def lstm_fe(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64", name="input_sentence")
    rule_mask = Input(shape=(None,), dtype="int64", name="rule_mask")

    # Feature extraction
    modified_input = Feature_extraction()([input_sentence, rule_mask])
    
    # Word embeddings 
    embedding = layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], 
                                embeddings_initializer=Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="word2vec")(modified_input)

    # Classifier Layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(embedding)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence, rule_mask], outputs=[out])    
    model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])

    return model

def bilstm_fe(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64", name="input_sentence")
    rule_mask = Input(shape=(None,), dtype="int64", name="rule_mask")

    # Feature extraction
    modified_input = Feature_extraction()([input_sentence, rule_mask])
    
    # Word embeddings 
    embedding = layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(modified_input)

    # Classifier Layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"]), name="classifier")(embedding)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence, rule_mask], outputs=[out])
    model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def bigru_fe(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64", name="input_sentence")
    rule_mask = Input(shape=(None,), dtype="int64", name="rule_mask")

    # Feature extraction
    modified_input = Feature_extraction()([input_sentence, rule_mask])
    
    # Word embeddings 
    embedding = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(modified_input)

    # Classifier Layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"]), name="classifier")(embedding)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence, rule_mask], outputs=[out])
    model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def gru_fe(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64", name="input_sentence")
    rule_mask = Input(shape=(None,), dtype="int64", name="rule_mask")
    
    # Feature extraction
    modified_input = Feature_extraction()([input_sentence, rule_mask])

    # Word embeddings 
    embedding = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(modified_input)

    # Classifier Layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(embedding)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence, rule_mask], outputs=[out])
    model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def birnn_fe(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64", name="input_sentence")
    rule_mask = Input(shape=(None,), dtype="int64", name="rule_mask")
    
    # Feature extraction
    modified_input = Feature_extraction()([input_sentence, rule_mask])

    # Word embeddings 
    embedding = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(modified_input)

    # Classifier Layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"]), name="classifier")(embedding)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence, rule_mask], outputs=[out])
    model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def rnn_fe(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64", name="input_sentence")
    rule_mask = Input(shape=(None,), dtype="int64", name="rule_mask")
    
    # Feature extraction
    modified_input = Feature_extraction()([input_sentence, rule_mask])

    # Word embeddings 
    embedding = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(modified_input)

    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(embedding)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence, rule_mask], outputs=[out])
    model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model