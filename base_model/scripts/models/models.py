import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
# from keras_bert import get_model

def rnn(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    # out = layers.SimpleRNN(config["hidden_units_classifier"], return_sequences = True, dropout=config["dropout"], name='classifier_1', trainable=True)(out)
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier_2")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def rnn_mask(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Mask layer
    mask_embedding = layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"], name='mask_embedder', trainable=True)(out)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[mask])    
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def birnn(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"]), name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def birnn_mask(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Mask layer
    mask_embedding = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_mask_embedder"], return_sequences = True, dropout=config["dropout"]), name='mask_embedder', trainable=True)(out)
    mask = layers.Dense(1, activation='sigmoid', name='mask', trainable=True)(mask_embedding)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[mask])    
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def lstm(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=False, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def bilstm(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"]), name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def bigru(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"]), name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def gru(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], 
                            word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

# def bert(config, word_vectors):

#     # Input sentence (padded and tokenized)
#     input_sentence = Input(shape=(None,), dtype="int64")
#     attention_mask = Input(shape=(None,), dtype="int64")
    
#     # Bert layer-by-layer implementation
#     bert_model = get_model(token_num=30000,
#                             head_num=12,
#                             transformer_num=12,
#                             embed_dim=768,
#                             feed_forward_dim=3072,
#                             seq_len=512,
#                             pos_num=512,
#                             dropout_rate=0.05,
#                             )

#     # Classifier Layer
#     out = layers.SimpleRNN(512, dropout=config["dropout"], name="classifier")(out)
#     out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
#     # The model
#     model = Model(inputs=[input_sentence, attention_mask], outputs=[out])
#     model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
#     return model