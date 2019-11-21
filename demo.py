import logging

import gensim
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, Embedding
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(name)-12s %(levelname)-5s] %(message)s')

logging.info("Loading data from sentences_ml.csv")
data = pd.read_csv("sentences_ml.csv")

# code output as two 'neurons': tone (0=negative, 1=positive) and subjectivity (0=no sentiment, 1=sentiment)
# can also keep single neuron with three values, or three neurons (pos/neut/neg)
# but in my test that performed slightly worse (but that could well depend on task)
labels = np.asarray([[(x + 1) / 2, int(x != 0)] for x in data.tone])

# tokenize the input
tokenizer = Tokenizer(num_words=10000)  # basically no length limit, these are headlines anyway
tokenizer.fit_on_texts(data.lemmata)
sequences = tokenizer.texts_to_sequences(data.lemmata)
tokens = pad_sequences(sequences)  # input sequences

# split train / test (in this case, on data.gold)
train_data = tokens[data.gold == 0]
test_data = tokens[data.gold == 1]

train_labels = labels[data.gold == 0]
test_labels = labels[data.gold == 1]

test_ids = data.id[data.gold == 1].values

logging.info("Loading embeddings")
# to download our Dutch embeddings: wget http://i.amcat.nl/w2v -O data/tmp/w2v_320d
# of course, many pretrained embeddings exist for many languages
embeddings = gensim.models.Word2Vec.load("/home/wva/ecosent/data/tmp/w2v_320d")
embeddings_matrix = np.zeros((len(tokenizer.word_index) + 1, embeddings.vector_size))
for word, i in tokenizer.word_index.items():
    if word in embeddings.wv:
        embeddings_matrix[i] = embeddings.wv[word]

logging.info("Creating model")
# embedding input layer
embedding_layer = Embedding(embeddings_matrix.shape[0],
                            embeddings_matrix.shape[1],
                            weights=[embeddings_matrix],
                            input_length=train_data.shape[1],
                            trainable=True)

sequence_input = Input(shape=(train_data.shape[1],), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

# Add convolution and pooling layers
x = Conv1D(128, 3, activation='relu')(embedded_sequences)
x = GlobalMaxPooling1D()(x)

# Add dense hidden layer(s)
x = Dense(64, activation='relu')(x)

# Add output layer
preds = Dense(2, activation='sigmoid')(x)

# Create and compile Model
model = Model(sequence_input, preds)
model.compile(loss='mean_absolute_error', optimizer=RMSprop(lr=0.004))

print(model.summary())
