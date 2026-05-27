#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ['I love my dog Miski','I love my cat Luna', 'You love my dog', 'So you think my dog is amazing']

n_words = 100
tokeniser = Tokenizer(n_words, oov_token = "<OOV>") #out of vocabulary token gets replaced by oov (out of vocabulary)
tokeniser.fit_on_texts(sentences)
word_index = tokeniser.word_index
sentences = tokeniser.texts_to_sequences(sentences)
padded = pad_sequences(sentences)

print(word_index)
print(sentences)
print(padded)