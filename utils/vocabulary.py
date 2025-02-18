import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
from nltk.tokenize import word_tokenize


#This class is used to represent an NLP vocabulary (unique set of words seen in some dataset)
#In this case, it's the unique set of words seen in the set of image captions
class Vocabulary(object):
    def __init__(self, size, save_file=None):
        #list of all words in the vocabulary
        self.words = []

        self.word2idx = {}

        #list of ints giving frequency of each word in vocabulary
        self.word_frequencies = []

        #size of the vocabulary
        self.size = 0

        #if loading from existing vocab CSV file, load it now
        if save_file is not None:
            self.load(save_file)


    def build(self, sentences):
        """ Build the vocabulary from the list of (all) captions passed, and compute the frequency of each word. """
        #print("in build() in vocabulary.py")

        #this var is a dict mapping words in the captions to their frequencies
        word_counts = {}

        #iterate over all captions
        for sentence in tqdm(sentences):
            #get list of words that occur in the sentence
            for w in word_tokenize(sentence.lower()):
                #increment word count if we see this word
                word_counts[w] = word_counts.get(w, 0) + 1.0


        #set vocab size now
        #CHANGED BY NWEINER on 12/14/22
        self.size = len(word_counts.keys())
        #print(word_counts.keys())

        #make sure size has been set appropriately
        assert self.size - 1 <= len(word_counts.keys())

        #insert start indicator strings at beginning of words and word2idx arrays
        self.words.append('<start>')
        self.word2idx['<start>'] = 0
        self.word_frequencies.append(1.0)

        #sort the word counts (ints describing freq of each word) in descending order
        word_counts = sorted(list(word_counts.items()), key=lambda x: x[1], reverse=True)

        #iterate over entire vocab
        for idx in range(self.size - 1):
            #get the word and its frequency
            word, frequency = word_counts[idx]

            #add this word to the vocab
            self.words.append(word)
            self.word2idx[word] = idx + 1
            self.word_frequencies.append(frequency)


        self.word_frequencies = np.array(self.word_frequencies)
        self.word_frequencies /= np.sum(self.word_frequencies)
        self.word_frequencies = np.log(self.word_frequencies)
        self.word_frequencies -= np.max(self.word_frequencies)



    def process_sentence(self, sentence):
        """ Tokenize a sentence, and translate each token into its index in the vocabulary. """
        #tokenize words (get list of all unique words in sentence)
        words = word_tokenize(sentence.lower())
        #print(words)

        #for each word, convert word to an index using our vocabulary
        word_idxs = [self.word2idx[w] for w in words]

        return word_idxs


    def get_sentence(self, idxs):
        """ Translate a vector of indicies into a sentence. """
        words = [self.words[i] for i in idxs]
        if words[-1] != '.':
            words.append('.')
        length = np.argmax(np.array(words)=='.') + 1
        words = words[:length]
        sentence = "".join([" "+w if not w.startswith("'") \
                            and w not in string.punctuation \
                            else w for w in words]).strip()
        return sentence


    def save(self, save_file):
        """ Save the vocabulary to a CSV file. """
        data = pd.DataFrame({'word': self.words, 'index': list(range(self.size)), 'frequency': self.word_frequencies})
        data.to_csv(save_file)


    def load(self, save_file):
        """ Load the vocabulary from a file. """
        assert os.path.exists(save_file)
        data = pd.read_csv(save_file)
        self.words = data['word'].values
        self.word2idx = {self.words[i]:i for i in range(self.size)}
        self.word_frequencies = data['frequency'].values
