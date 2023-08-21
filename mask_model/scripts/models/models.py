import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Multiply(keras.layers.Layer):
    def __init__(self, name="multiply"):
        super(Multiply, self).__init__(name="multiply")
        self.supports_masking = True

    def call(self, inputs):
        output = layers.Multiply()(inputs)
        return output

#### BiLSTM mask ####
def lstm_bilstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bilstm_bilstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bigru_bilstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def gru_bilstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def birnn_bilstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def rnn_bilstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

###### BiGRU mask ######
def rnn_bigru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def birnn_bigru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def gru_bigru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bigru_bigru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def lstm_bigru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bilstm_bigru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

###### LSTM mask ######
def rnn_lstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def birnn_lstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def gru_lstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bigru_lstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def lstm_lstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bilstm_lstm_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.LSTM(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

###### RNN mask ######
def rnn_rnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def birnn_rnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def gru_rnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bigru_rnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def lstm_rnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bilstm_rnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

###### GRU mask ######
def rnn_gru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def birnn_gru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def gru_gru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bigru_gru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def lstm_gru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bilstm_gru_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.GRU(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

###### BiRNN mask ######
def rnn_birnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def birnn_birnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def gru_birnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bigru_birnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def lstm_birnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"])(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

def bilstm_birnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model

# BERT mask
def bilstm_birnn_mask(config, word_vectors):
    
    # Input sentence
    input_sentence = keras.Input(shape=(None,), dtype="int64")
    
    # Word embeddings
    embedding = layers.Embedding(word_vectors.shape[0], 
                                word_vectors.shape[1], 
                                embeddings_initializer=keras.initializers.Constant(word_vectors), 
                                trainable=config["fine_tune_embedding_model"], 
                                mask_zero=True, 
                                name="Word2vec")(input_sentence)
    
    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True))(embedding)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)

    # Applying Mask
    multiply_emb = Multiply()([embedding, mask])
    
    # Classifier layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"]))(multiply_emb)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = keras.models.Model(inputs=[input_sentence], outputs=[out, mask])    
    
    return model