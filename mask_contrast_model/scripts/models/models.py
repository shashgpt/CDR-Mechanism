import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()



class Add(keras.layers.Layer):
    def __init__(self, name="add", **kwargs):
        super(Add, self).__init__(name="add", **kwargs)
        self.supports_masking = True
        self.add = layers.Add()
    
    def call(self, inputs):
        output = self.add(inputs)
        return output

class Multiply(keras.layers.Layer):
    def __init__(self, name="multiply", **kwargs):
        super(Multiply, self).__init__(name="multiply", **kwargs)
        self.supports_masking = True
        self.multiply_emb = layers.Multiply()

    def call(self, inputs):
        output = self.multiply_emb(inputs)
        return output

class Binomial(keras.layers.Layer):
    def __init__(self, name="binomial", **kwargs):
        super(Binomial, self).__init__(name="binomial", **kwargs)
        self.supports_masking = True
        self.add = Add()
        self.multiply = Multiply()
    
    def call(self, inputs): # inputs[1]=> contrast_prob, inputs[0]=> mask_prob, 0=> no contrast, 1=> contrast, Binomial = (1-P(C)) + P(C)*P(M)
        out_2 = tf.math.subtract(1.0, inputs[1])
        out_3 = self.multiply([inputs[0], inputs[1]])
        output = self.add([out_2, out_3])
        return output

#### BiLSTM mask, BiLSTM contrast ####
def lstm_bilstm_mask_bilstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bilstm_bilstm_mask_bilstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_bilstm_mask_bilstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_bilstm_mask_bilstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def birnn_bilstm_mask_bilstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def rnn_bilstm_mask_bilstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

###### BiGRU mask, BiGRU contrast ######

def lstm_bigru_mask_bigru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bilstm_bigru_mask_bigru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_bigru_mask_bigru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_bigru_mask_bigru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def rnn_bigru_mask_bigru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def birnn_bigru_mask_bigru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True), name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True), name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

###### RNN mask, RNN contrast ######

def rnn_rnn_mask_rnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def birnn_rnn_mask_rnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def lstm_rnn_mask_rnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bilstm_rnn_mask_rnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_rnn_mask_rnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_rnn_mask_rnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

###### LSTM mask, LSTM contrast ######
def rnn_lstm_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def birnn_lstm_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def lstm_lstm_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bilstm_lstm_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_lstm_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_lstm_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

###### GRU mask, GRU contrast ######
def rnn_gru_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def birnn_gru_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def lstm_gru_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bilstm_gru_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_gru_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])    
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])

    return model

def bigru_gru_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder")(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

###### BiRNN mask, BiRNN contrast ######
def rnn_birnn_mask_birnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def birnn_birnn_mask_birnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def lstm_birnn_mask_birnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bilstm_birnn_mask_birnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_birnn_mask_birnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_birnn_mask_birnn_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

#### Extra combinations #####
def gru_bigru_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_bigru_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_bigru_mask_bilstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_bilstm_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_bilstm_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def gru_bilstm_mask_bigru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_bigru_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_bigru_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_bigru_mask_bilstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_bilstm_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_bilstm_mask_bigru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def bigru_bilstm_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def lstm_bigru_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def lstm_bigru_mask_lstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def lstm_bigru_mask_bilstm_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.Bidirectional(layers.LSTM(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder"))(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model

def lstm_bilstm_mask_gru_contrast(config, word_vectors):

    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                name="word2vec", 
                                mask_zero=True)(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], trainable=True, name="Mask_embedder"))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # Contrast layer
    contrast_embeddings = layers.GRU(config["hidden_units_contrast_embedder"], dropout=config["dropout"], trainable=True, name="Contrast_embedder")(embedding)
    contrast = layers.Dense(1, activation='sigmoid', name='contrast', trainable=True)(contrast_embeddings)
    
    # Add contrast with mask and clip it
    modified_mask = Binomial()([mask, contrast])
    
    # Applying Mask
    multiply_emb = Multiply()([embedding, modified_mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, modified_mask, contrast])
    model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
                  loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])    
    
    return model