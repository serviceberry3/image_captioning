__author__ = 'tylin'
__version__ = '2.0'
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>segToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon
import urllib
import copy
import itertools
import os
import string

from tqdm import tqdm
from nltk.tokenize import word_tokenize

#CHANGE by NWEINER on 12/14/22:
#we'll trim the amount of data we're using from COCO
NUM_DATA = 82787

#specify whether to pull images from local disk, or use COCO api to fetch them via URLs
LOCAL = True

class COCO:
    def __init__(self, config, num_data, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        
        #if using COCO:
        #dicts to keep track of annotation data (see createIndex() function for more info)
        self.dataset = {}

        self.annId_to_ann = {}
        self.imgId_to_ann = {}
        self.catToImgs = {}
        self.imgId_to_img = {}
        self.cats = {}
        self.imgName_to_id = {}

        #if using SBU:
        self.imgId_to_cap = {}
        self.imgIds = []

        #list of the image IDs we want to use
        self.list_of_img_ids_were_using = []

        #save the configuration as object property
        self.config = config


        #if a JSON file is passed that contains the image captions, load them into memory (into the #dataset variable)
        if not annotation_file == None:
            print('Loading annotations into memory...')

            #load the annotations file, and record how long loading annotations takes
            tic = time.time()

            #json.load() takes the file and returns a JSON object, which contains data in form of key/value pairs
            dataset = json.load(open(annotation_file, 'r'))
            print('Finished loading', annotation_file, 'in %0.2fs'%(time.time()- tic))

            #save the JSON object into the "dataset" property of this COCO object
            self.dataset = dataset

            '''
            if 'annotations' in self.dataset:
                #print("found 'annotations' section in the JSON")
                self.dataset['annotations'] = self.dataset['annotations']
                #self.dataset['annotations'] = self.dataset['annotations'][:NUM_DATA]
                #print(self.dataset['annotations'])

            if 'images' in self.dataset:
                #print("found 'images' section in the JSON")
                self.dataset['images'] = self.dataset['images']
                #print(self.dataset['images'])

            if 'categories' in self.dataset:
                print("found 'categories' section in the JSON")
                #self.dataset['categories'] = self.dataset['categories'][:NUM_DATA]
            '''

            #this function changes captions to all lowercase
            self.process_dataset()

            self.createIndex(False)


    #using the JSON object extracted from the COCO annotations file, create some dicts to organize the data into an "index"
    #NOTE: each unique-id-image can have multiple annotations, each annotation having its own unique annotation ID and its own unique caption
    def createIndex(self, force_coco):
        print('Loading JSON data into convenient dicts via createIndex()...')

        ctr = 0

        if (self.config.dataset == 'coco' or force_coco):

            #COCO json has various sections. images section contains image IDs, license info, file name info, URL for img, etc.
            #annotations section contains image id and metadata about the image, including its caption, area, etc.

            #dict that maps annotation ID to the annotation (a dict) itself
            annId_to_ann = {}

            #dict that maps image ID to the image annotation (a dict)
            imgId_to_ann = {}


            catToImgs = {}
            cats = {}

            #dict that maps image ID to image metadata
            imgId_to_img = {}

            #dict that maps [either image URL or image filename] to the image ID
            imgName_to_id = {}

            #if the JSON object contains "annotations" section
            if 'annotations' in self.dataset:
                #for each annotation
                for ann in self.dataset['annotations']:
                    this_img_id = ann['image_id']

                    imgId_to_ann[this_img_id] = []

                    #create an entry in the dict that maps image ID to the annotation dict for that image, and populate entry key with the image ID
                    #this dict maps image id to the annotation dict for that image. populate the value of each entry with the annotation
                    imgId_to_ann[this_img_id] += [ann]

                    #create entry in dict that maps annotation ID to the annotation dict for that image, and populate entry key with the annotation ID
                    #this dict maps annotation id to the annotation dict for that image. populate the value of each entry with the annotation
                    annId_to_ann[ann['id']] = ann

                    #if we're loading images via COCO URLs, append the first num_data images to the image id list
                    if (self.config.local == False and ctr < num_data): 
                        self.list_of_img_ids_were_using.append(this_img_id)

                    ctr += 1

                    '''
                    if (ann['image_id'] == 673):
                        print("imgid_to_ann entry is", imgId_to_ann[ann['image_id']])
                        print("annid_to_ann entry is", annId_to_ann[ann['id']])
                    '''

                #this is another way to do this (with one-line codes)
                #imgId_to_ann = {ann['image_id']: [] for ann in self.dataset['annotations']}
                #annId_to_ann = {ann['id']: [] for ann in self.dataset['annotations']}


            #if the JSON object contains "images" section
            if 'images' in self.dataset:
                #get list of all image IDs, populating only keys with the image id
                imgId_to_img = {im['id']: {} for im in self.dataset['images']}

                #for each image dict
                for img in self.dataset['images']:
                    #the #imgs var maps image id to the dictionary containing rest of image data
                    imgId_to_img[img['id']] = img

                    #the #img_name_to_id var maps image coco URL or file name to image id
                    #CHANGE BY NWEINER: instead of using file_name, I'll use coco url
                    if self.config.local:
                        imgName_to_id[img['file_name']] = img['id']
                    else:
                        imgName_to_id[img['coco_url']] = img['id']


            #if the JSON object contains "categories" section
            if 'categories' in self.dataset:
                cats = {cat['id']: [] for cat in self.dataset['categories']}

                for cat in self.dataset['categories']:
                    cats[cat['id']] = cat

                catToImgs = {cat['id']: [] for cat in self.dataset['categories']}

                for ann in self.dataset['annotations']:
                    catToImgs[ann['category_id']] += [ann['image_id']]


            #save the indices (dicts) we've created as class properties

            #maps annotation id to annotation dict
            self.annId_to_ann = annId_to_ann

            #maps image id to annotation dict
            self.imgId_to_ann = imgId_to_ann

            self.catToImgs = catToImgs

            #maps image id to dict containing image data
            self.imgId_to_img = imgId_to_img

            self.cats = cats

            #remember, I changed this on 12/14/22 to optionally map the COCO url for the image to the id of that image, instead of the local filename
            self.imgName_to_id = imgName_to_id


        #the sbu json only contains the image urls and their respective captions
        elif self.config.dataset == 'sbu':
            imgId_to_cap = {}
            imgIds = []
            print("createIndex(): iterating over sbu image IDs and creating image to caption dict")

            #for each annotation
            for url in self.dataset['image_urls']:
                #extract the 10-digit img ID from end of url
                id_from_url = url[-14:-4]
                #print("createIndex(): adding sbu img id {} with its respective caption to imgId_to_cap".format(id_from_url))

                imgId_to_cap[id_from_url] = self.dataset['captions'][ctr]
                imgIds.append(id_from_url)

                ctr += 1


            self.imgId_to_cap = imgId_to_cap
            self.imgIds = imgIds


        print('Index creation complete.')



    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('%s: %s'%(key, value))



    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                # this can be changed by defaultdict
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]

        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids


    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids


    def getImgIds(self, imgIds = [], catIds = []):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given idsimgToAnns
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        #if no category or img id filters are passed, just return list of IDs of images we're using
        if len(imgIds) == len(catIds) == 0:
            #ids = self.imgs.keys()
            ids = self.list_of_img_ids_were_using

        else:
            ids = set(imgIds)

            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])


        return list(ids)


    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]


    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]


    #load the data about the image with the passed id(s), including COCO url for that img
    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            print("ids is a list")
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            print("loadImgs: id is", ids)
            return [self.imgs[ids]]


    def loadRes(self, config):
        """
        Load result file and return a result api object (a COCO object).
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO(config, config.num_val_data)

        resFile = config.eval_result_file

        if config.dataset == 'coco':
            res.dataset['images'] = [img for img in self.dataset['images']]
            # res.dataset['info'] = copy.deepcopy(self.dataset['info'])
            # res.dataset['licenses'] = copy.deepcopy(self.dataset['licenses'])

        print('Loading and preparing results... ')
        tic = time.time()

        #resfile by default is ie val/results.json for evaluation
        #load in the json data that was saved during eval() 
        anns = json.load(open(resFile))
        #print("anns is", anns)

        assert type(anns) == list, 'results in not an array of objects'

        #extract image IDs from the json
        annsImgIds = [ann['image_id'] for ann in anns]

        #print("annsImgIds is", annsImgIds)
        #print("self.getImgIds is", self.getImgIds())
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), 'Results do not correspond to current coco set'

        assert 'caption' in anns[0]

        if config.dataset == 'coco':
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])

            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        elif config.dataset == 'sbu':
            pass
            #imgIds = set(res.imgIds) & set([ann['image_id'] for ann in anns])
            #res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]

            #res.imgId

        for id, ann in enumerate(anns):
            ann['id'] = id + 1

        print('DONE (t=%0.2fs)'%(time.time()- tic))

        res.dataset['annotations'] = anns

        res.createIndex(True)

        return res



    def download( self, tarDir = None, imgIds = [] ):
        '''
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        '''
        if tarDir is None:
            print('Please specify target directory')
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)

        if not os.path.exists(tarDir):
            os.makedirs(tarDir)


        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img['file_name'])
            if not os.path.exists(fname):
                urllib.urlretrieve(img['coco_url'], fname)
            print('downloaded %d/%d images (t=%.1fs)'%(i, N, time.time()- tic))


    #process the json annotations file for this COCO object
    def process_dataset(self):
        if self.config.dataset == 'sbu':
            print("processing sbu captions...")
            for caption in self.dataset['captions']:
                q = caption.lower()

                #make sure caption ends with a sentence
                if q[-1] != '.':
                    q = q + '.'

                #save the new modded caption
                caption = q

        elif self.config.dataset == 'coco':
            #for each json entry in the "annotations" section of the JSON file
            for ann in self.dataset['annotations']:
                #change caption to all lowercase
                q = ann['caption'].lower()

                #make sure caption ends with a sentence
                if q[-1] != '.':
                    q = q + '.'

                #save the new modded caption
                ann['caption'] = q


    #filter the image captions by length
    def filter_by_cap_len(self, max_cap_len):
        print("Filtering the captions by length...")

        if (self.config.dataset == 'coco'):
            keep_ann = {}
            keep_img = {}


            for ann in tqdm(self.dataset['annotations']):
                if len(word_tokenize(ann['caption'])) <= max_cap_len:
                    keep_ann[ann['id']] = keep_ann.get(ann['id'], 0) + 1
                    keep_img[ann['image_id']] = keep_img.get(ann['image_id'], 0) + 1


            self.dataset['annotations'] = \
                [ann for ann in self.dataset['annotations'] \
                if keep_ann.get(ann['id'], 0) > 0]

            self.dataset['images'] = \
                [img for img in self.dataset['images'] \
                if keep_img.get(img['id'], 0) > 0]


        elif (self.config.dataset == 'sbu'):
            keep_cap = {}

            ctr = 0
            for cap in tqdm(self.dataset['captions']):
                #make sure the len of this caption is less than max len. if it's not, we'll throw it out
                if len(word_tokenize(cap)) <= max_cap_len:
                    #make note to keep this caption
                    keep_cap[self.imgIds[ctr]] = keep_cap.get(self.imgIds[ctr], 0) + 1
                else:
                    if self.imgIds[ctr] == '240d5ff280':
                        print("PRUNED OUT in filter by cap")
                    if self.imgIds[ctr] in self.list_of_img_ids_were_using:
                        self.list_of_img_ids_were_using.remove(self.imgIds[ctr])

                ctr += 1

            #prune the image urls first
            ctr = 0
            new_img_urls = []
            for url in self.dataset['image_urls']:
                if keep_cap.get(self.imgIds[ctr], 0) > 0:
                    new_img_urls.append(url)
                ctr += 1
            self.dataset['image_urls'] = new_img_urls


            #then prune the captions
            ctr = 0
            new_captions = []
            for cap in self.dataset['captions']:
                if keep_cap.get(self.imgIds[ctr], 0) > 0:
                    new_captions.append(cap)
                ctr += 1
            self.dataset['captions'] = new_captions


        #remake the index after modifying the data
        self.createIndex(False)



    def filter_by_words(self, vocab):
        print("Filtering the captions by words...")

        if (self.config.dataset == 'coco'):
            keep_ann = {}
            keep_img = {}

            #for all of the annotations in the JSON
            for ann in tqdm(self.dataset['annotations']):
                keep_ann[ann['id']] = 1

                #get list of words in this caption
                words_in_ann = word_tokenize(ann['caption'])


                #check whether this caption contains a word that is not in the vocabulary. if it does, flag the annotation to be removed completely
                for word in words_in_ann:
                    if word not in vocab:
                        #print("WORD {} NOT IN VOCAB".format(word))
                        keep_ann[ann['id']] = 0
                        break

                #keep_img will be 1 by default
                keep_img[ann['image_id']] = keep_img.get(ann['image_id'], 0) + 1


            #prune some annotations (which correspond to captions, in this case) out of the annotations set
            self.dataset['annotations'] = \
                [ann for ann in self.dataset['annotations'] \
                if keep_ann.get(ann['id'], 0) > 0]

            #also prune images
            self.dataset['images'] = \
                [img for img in self.dataset['images'] \
                if keep_img.get(img['id'], 0) > 0]

        elif (self.config.dataset == 'sbu'):
            keep_cap = {}

            ctr = 0
            for cap in tqdm(self.dataset['captions']):
                #default to keeping the caption
                keep_cap[self.imgIds[ctr]] = 1

                #get list of words in this caption
                words_in_ann = word_tokenize(cap)

                #check whether this caption contains a word that is not in the vocabulary. if it does, flag the annotation to be removed completely
                for word in words_in_ann:
                    if word not in vocab:
                        if self.imgIds[ctr] == '240d5ff280':
                            print("PRUNED OUT in filter_by_words")

                        if self.imgIds[ctr] in self.list_of_img_ids_were_using:
                            self.list_of_img_ids_were_using.remove(self.imgIds[ctr])

                        #print("WORD {} NOT IN VOCAB".format(word))
                        keep_cap[self.imgIds[ctr]] = 0
                        break

                ctr += 1

            
            #prune the image urls first
            ctr = 0
            new_img_urls = []
            for url in self.dataset['image_urls']:
                if keep_cap.get(self.imgIds[ctr], 0) > 0:
                    new_img_urls.append(url)
                ctr += 1
            self.dataset['image_urls'] = new_img_urls


            #then prune the captions
            ctr = 0
            new_captions = []
            for cap in self.dataset['captions']:
                if keep_cap.get(self.imgIds[ctr], 0) > 0:
                    new_captions.append(cap)
                ctr += 1
            self.dataset['captions'] = new_captions




        #remake the index
        self.createIndex(False)



    #return list of all image captions
    def all_captions(self):
        if self.config.dataset == 'coco':
            return [ann['caption'] for ann_id, ann in self.annId_to_ann.items() if ann['image_id'] in self.list_of_img_ids_were_using]

        #sbu version
        return [self.imgId_to_cap[id] for id in self.list_of_img_ids_were_using]
