import os
import math
import numpy as np

#pandas pkg is for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.
import pandas as pd
from tqdm import tqdm

from utils.coco.coco import COCO as myCOCO
from utils.vocabulary import Vocabulary

from pycocotools.coco import COCO

import skimage.io as io

import pylab

import matplotlib.pyplot as plt

#set this to True if you want to draw from images stored locally on disk. else will use COCO API to fetch images via URL
LOCAL = True

'''
Class that serves as a container for the dataset.
'''
class DataSet(object):
    def __init__(self,
                 image_ids, #ids of images
                 image_links,  #COCO link for each image (NWEINER CHANGED)
                 batch_size, #training batch size??
                 word_idxs=None, #these are the captions, converted into lists of indices (corresponding to index into the word vocabulary)
                 masks=None,
                 is_train=False,
                 shuffle=False):

        
        self.image_ids = np.array(image_ids)
        self.image_links = np.array(image_links)
        self.word_idxs = np.array(word_idxs)
        self.masks = np.array(masks)
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.image_ids)
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()

        if self.has_full_next_batch():
            start, end = self.current_idx, self.current_idx + self.batch_size
            current_idxs = self.idxs[start:end]
        else:
            start, end = self.current_idx, self.count
            current_idxs = self.idxs[start:end] + list(np.random.choice(self.count, self.fake_count))

        image_links = self.image_links[current_idxs]
        image_ids = self.image_ids[current_idxs]

        #if we're in training, return image ids, image links, captions, and masks
        if self.is_train:
            word_idxs = self.word_idxs[current_idxs]
            masks = self.masks[current_idxs]
            self.current_idx += self.batch_size

            return image_ids, image_links, word_idxs, masks
        else:
            self.current_idx += self.batch_size
            return image_links

    def has_next_batch(self):
        """ Determine whether there is a batch left. """
        return self.current_idx < self.count

    def has_full_next_batch(self):
        """ Determine whether there is a full batch left. """
        return self.current_idx + self.batch_size <= self.count


#prepare the training data
def prepare_train_data(config):
    """ Prepare the data for training the model. """
    print("Running prepare_train_data()...")

    #instantiate a COCO Python object, passing the JSON annotation file containing all the captions for all training images
    coco = myCOCO(config.train_caption_file)

    #pull caption for each image into a list
    #captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]

    #get the list of all files and directories
    path = "../image_captioning/train/images"
    files_list = os.listdir(path)

    #init empty list to hold all IDs of images stored locally
    ids = []

    #iterate through list of filenames
    for file in files_list:
        #make sure this is not a JSON file
        if (file[-3:] != 'son'):
            #extract the annotation ID corresponding to the jpg file, and append it to the running list of found IDs
            ids.append(int(file[20:27]))

    coco.list_of_img_ids_were_using = ids

    #filter by caption lengths
    coco.filter_by_cap_len(config.max_caption_length)

    #build vocabulary. in NLP, a vocabulary is the set of unkique words used to train a model
    print("Building the vocabulary...")
    vocabulary = Vocabulary(config.vocabulary_size)

    #if the file './vocabulary.csv' doesn't already exist, build it now and save it
    if not os.path.exists(config.vocabulary_file):
        vocabulary.build(coco.all_captions())
        vocabulary.save(config.vocabulary_file)

    #otherwise vocab CSV file (file containing all words found in captions) already exists, so just load it now
    else:
        vocabulary.load(config.vocabulary_file)
        
    print("Vocabulary has been built or loaded from ./vocabulary.csv")
    print("Number of words = %d" %(vocabulary.size))

    #filter captions by words
    #set is like a Python list but can have mixed datatypes
    coco.filter_by_words(set(vocabulary.words))

    #process image captionsa
    print("Processing the captions (converting them to lists of word indices)...")

    #check if file './train/anns.csv' already exists

    #if not, then create this csv now
    if not os.path.exists(config.temp_annotation_file):
        image_ids = ids
        print("{} training images were found locally, in {}".format(len(image_ids), path))

        #pull id for each image into a list
        #image_ids = [coco.annId_to_ann[ann_id]['image_id'] for ann_id in coco.annId_to_ann]
        #print("found following image ids locally:", ids)

        #extract captions for the images that are stored locally
        captions = [coco.imgId_to_ann[image_id][0]['caption'] for image_id in image_ids]

        #CHANGED BY NWEINER 12/14/22: 
        if LOCAL:
            image_files = [os.path.join(config.train_image_dir, coco.imgId_to_img[image_id]['file_name']) for image_id in image_ids]

            #create pandas dataframe with three cols: image id, the image's local filename, and the image's caption
            annotations = pd.DataFrame({'image_id': image_ids, 'image_file': image_files, 'caption': captions}) 

        #instead of using image filenames, use the COCO link for the image (to avoid storing tons of images locally)
        else:
            image_links = [coco.imgId_to_img[image_id]['coco_url'] for image_id in image_ids] #recall that coco.imgs maps image id to a dict containing image data

            #creata a pandas dataframe with three cols: image id, the image's COCO url, and the image's caption
            annotations = pd.DataFrame({'image_id': image_ids, 'image_link': image_links, 'caption': captions}) #CHANGED BY NWEINER on 12/14/22: change image_file to image_link

        #print("dataframe created is", annotations)
        

        #save the annotations in the file './train/anns.csv'
        annotations.to_csv(config.temp_annotation_file)


    #otherwise we can simply read from the existing annotations csv
    else:
        annotations = pd.read_csv(config.temp_annotation_file)

        #extract all image captions, image ids, and image filenames from the annotations JSON
        captions = annotations['caption'].values
        image_ids = annotations['image_id'].values

        if LOCAL:
            image_files = annotations['image_file'].values
        else:
            image_links = annotations['image_link'].values


    #check if the training file  ./train/data.npy' already exists
    if not os.path.exists(config.temp_data_file):
        word_idxs = []
        masks = []

        #if the file doesn't exist, create the word indices now

        #run progress bar as iterate over all captions
        for caption in tqdm(captions):
            #print("processing caption '", caption, "'")

            #turn the captions into a list of word indices
            current_word_idxs_ = vocabulary.process_sentence(caption)
            current_num_words = len(current_word_idxs_)

            #initialize two arrays of zeros the length of the max caption
            current_word_idxs = np.zeros(config.max_caption_length, dtype = np.int32)
            current_masks = np.zeros(config.max_caption_length)

            #fill the word indices array with the indices of words in sequence of this caption
            current_word_idxs[:current_num_words] = np.array(current_word_idxs_)

            #fill masks array with 1s up thru num of words of the caption
            current_masks[:current_num_words] = 1.0

            #append the word indices and masks array to the list of all word indices arrays
            word_idxs.append(current_word_idxs)
            masks.append(current_masks)


        word_idxs = np.array(word_idxs)
        masks = np.array(masks)

        #save the word indices array and masks array into the data.npy file as a dict
        data = {'word_idxs': word_idxs, 'masks': masks}
        np.save(config.temp_data_file, data)

    #if the file already exists, load the training data
    else:
        #load the temporary data file
        data = np.load(config.temp_data_file, allow_pickle=True).item()

        word_idxs = data['word_idxs']
        masks = data['masks']


    print("Image captions have been processed.")
    print("Number of captions = %d" %(len(captions)))

    print("Building the DataSet object using the images and digit-ified captions...")

    #instantiate a DataSet object with these image IDs, image links, the appropriate batch size, and the word indices arrays (captions)
    if LOCAL:
        dataset = DataSet(image_ids, image_files, config.batch_size, word_idxs, masks, True, True)
    else:
        dataset = DataSet(image_ids, image_links, config.batch_size, word_idxs, masks, True, True)
    print("Dataset built.")


    #return the DataSet object
    return dataset


#prepare evaluation data
def prepare_eval_data(config):
    """ Prepare the data for evaluating the model. """
    coco = myCOCO(config.eval_caption_file)
    image_ids = list(coco.imgs.keys())
    image_files = [os.path.join(config.eval_image_dir, coco.imgs[image_id]['file_name']) for image_id in image_ids]

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_size, config.vocabulary_file)
    else:
        vocabulary = build_vocabulary(config)

    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    print("Building the dataset...")
    dataset = DataSet(image_ids, image_files, config.batch_size)
    print("Dataset built.")
    return coco, dataset, vocabulary



#prepare the test data
def prepare_test_data(config):
    """ Prepare the data for testing the model. """
    files = os.listdir(config.test_image_dir)
    image_files = [os.path.join(config.test_image_dir, f) for f in files if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    image_ids = list(range(len(image_files)))

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_size, config.vocabulary_file)
    else:
        vocabulary = build_vocabulary(config)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    print("Building the dataset...")
    dataset = DataSet(image_ids, image_files, config.batch_size)
    print("Dataset built.")
    return dataset, vocabulary



def build_vocabulary(config):
    """ Build the vocabulary from the training data and save it to a file. """
    coco = myCOCO(config.train_caption_file)
    coco.filter_by_cap_len(config.max_caption_length)

    vocabulary = Vocabulary(config.vocabulary_size)
    vocabulary.build(coco.all_captions())
    vocabulary.save(config.vocabulary_file)
    return vocabulary
