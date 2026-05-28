#!/usr/bin/env python3

import json
import tensorflow as tf  #getting access to the funtions that can create neural networks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import io

sentences = []
labels = []
urls = []

with open("archive/sarcasm.json", 'r') as f:
    for line in f:
        item = json.loads(line)
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])

vocab_size = 10000            # working with the 10000 most frequent words to keep
embedding_dim = 16            # getting 16 coordinates in the 16 multidimention space, the more dimentions, there better representation there can be
max_length = 100              # max length of tokenised sequences (sentences)
trunc_type='post'             # if surpassing the max length then truncate on post
padding_type='post'           # if the length is less add 0 to the end
oov_tok = "<OOV>"             # if there is word that is not recognisable because it is not in the test data, set the out of vocabulary token and replace the words it does not recognise with this out of vocabulary token
training_size = 20000         # train with 1st 20000 sentences, the test sentences are the remaining

# dividing the training sentences and testing sentences
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

#create Tokenizer instance to tokenise the sentences (as in assigning them a number)
tokeniser = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokeniser.fit_on_texts(training_sentences)  #feed the tokeniser on the training sentences, feed ONLY on training sentences (this makes the neural work with only that knowledge base of words)
#performing the tokenisation
word_index = tokeniser.word_index

#creating sequences of the training words (lists of tokens (word number) that build a word)
training_sequences = tokeniser.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)  #adding 0 as padding to fit the max len
#creating sequences of the testing words 
testing_sequences = tokeniser.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type) #adding 0 as padding to fit the max len

##understand everything up till here
# turning regular lists into numpy arrays that support matrix arithmetic
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_length,)),                                      #signaling any input is containing 100 numbers (from padding)
    tf.keras.layers.Embedding(vocab_size, embedding_dim),                            #embedding (representing words): turning a word into coordinates for a multidimention for the neural network (any available coordinate at the moment, they later change)
    tf.keras.layers.GlobalAveragePooling1D(),                                        #average: average all the word vectors in the sentence and produce one vector that represents the entire sentece
    tf.keras.layers.Dense(24, activation='relu'),                                    #creating a layer of 24 neurons. Each neuron specialises in something, relu only keeps positive coordinates  (each neuron has random weights)
    tf.keras.layers.Dense(1, activation='sigmoid')                                   #creating a layer of 1 neuron and connect them to the layer before. (Only one becuase only one solution is needed for sarcastic or not sarcastic). sigmoid convertes any number into a number from 0 to 1 
])
#neural networks training set up
model.compile(loss='binary_crossentropy', # prediction of accuracy for the model, use binary classification problem because sarcasm is binary 1 (sarcastis) or 0 (not sarcastic), model might predict 0.95 for 1 (good prediction), this error measurement helps update the neuron weight
              optimizer='adam',           # use adam method to update the weight of each neuron,
              metrics = ['accuracy'])     # metrics shows the percentage of predictions it gets right during training

model.summary()                           #showing summary of the the layers that exist, the layer outputs and the weights the network is learning to change                                      



num_epochs = 30 # epochs means the number of times the model passes through the entire training data set, 30 times
#TRAINING
# start training the model, store all training statistics inside history 
history = model.fit(training_padded,    # 20,000 headlines wirh 100 integers each (the length of words of each sentence is 100)
                    training_labels,    # the correct answers for each headline (1 or 0, sarcastic or not)
                    epochs=num_epochs,  # train 30 times, 
                    validation_data=(testing_padded, testing_labels), #after each epoch the model tests itself on unseen headlines
                    verbose=2)          # determines what is shown in the terminal for each epochs (accuracy (how correct), loss (how wrong), (training data) val_accuracy, val_los (test data))

# draw a graph using values saved in history
def plot_graphs(history, string):
    plt.plot(history.history[string])        # plot training metric
    plt.plot(history.history['val_'+string]) # plot validation testing metric
    plt.xlabel("Epochs")                     # label of axis
    plt.ylabel(string)                       # label of axis
    plt.legend([string, 'val_'+string])      # THIS?
    plt.show()                               # THIS?
  
# show how model improved across epochs
plot_graphs(history, "accuracy")            # plot the accuracy in training data and test data 
plot_graphs(history, "loss")                # plot the inaccuracy of training data and test data

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # instead of mapping from word to number we map from number to word

# turn sequences back into sentences, if word is not in knowledge base, use ?
def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_sentence(training_padded[0])) #print the 1st headline after converting numbers back into words
print(training_sentences[2])               # print original headline and its label
print(labels[2])

e = model.layers[0]                        # get model at layer 0 (embedding)
weights = e.get_weights()[0]               # weights contains the learned word coordinates
print(weights.shape)                       # shape will be (vocab_size (10000) , embedding_dim (16 as in 16 coordinates))

# create files for embeddings
out_v = io.open('vecs.tsv', 'w', encoding='utf-8') # store coodinates of words
out_m = io.open('meta.tsv', 'w', encoding='utf-8') # store words
for word_num in range(1, vocab_size):
    word = reverse_word_index.get(word_num, "")    # get the word from its mapped number
    embeddings = weights[word_num]                 # get the weights of the layer embeddings that represent the coordinates of the words
    out_m.write(word + "\n")                       # write word to meta.tsv
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n") # write coordinates to vecs.tsv
out_v.close()  
out_m.close()

#FINAL TESTING PREDICTION
#test the neural network with brand new, unknown sentences
sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
sequences = tokeniser.texts_to_sequences(sentence) # turn the word into numbers and create sequences (lists) for the sentences
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type) #pad the sequences with 0s (max len 100)
print(model.predict(padded)) # asking the model how sarcastuc are these sentences 

'''Validation/testing during training was done inside model.fit.

Final prediction testing is this last block with your two custom sentences.'''