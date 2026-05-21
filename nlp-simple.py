#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ['I love my dog Miski','I love my cat Luna']

#max number of most frequent words to take into account for the model
n_words = 100
tokeniser = Tokenizer(n_words)
#feed the tokeniser all the sentences
tokeniser.fit_on_texts(sentences)
#the list of words is available as the tokeniser index property
word_index = tokeniser.word_index
print(word_index)