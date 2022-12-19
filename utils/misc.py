
import numpy as np
import cv2
import heapq

#import coco api
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

import skimage.io as io

class ImageLoader(object):
    def __init__(self, mean_file):
        self.bgr = True
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)


    #function that loads images from a COCO link and preprocesses it
    #CHANGED BY NWEINER on 12/14/22: load imgs from COCO api instead of disk
    def load_image(self, image_link, coco_instance, local):
        """ Load and preprocess an image. """

        #print("loading img {}".format(image_link))

        if local:
            image = cv2.imread(image_link)
        else:
            #NWEINER CHANGED: instead of reading image from disk, if LOCAL is False, get it from coco api
            #open the image using io from skimage
            image = io.imread(image_link)

        #print("load_image trying to load COCO image link", image_link)

        if self.bgr:
            temp = image.swapaxes(0, 2)
            temp = temp[::-1]
            image = temp.swapaxes(0, 2)

        image = cv2.resize(image, (self.scale_shape[0], self.scale_shape[1]))

        offset = (self.scale_shape - self.crop_shape) / 2

        offset = offset.astype(np.int32)

        image = image[offset[0]:offset[0]+self.crop_shape[0], offset[1]:offset[1]+self.crop_shape[1]]

        image = image - self.mean

        return image


    #CHANGED BY NWEINER on 12/14/22: load via image links, not files
    def load_images(self, image_links, coco_instance, local):
        """ Load and preprocess a list of images. """
        images = []

        #CHANGED BY NWEINER on 12/14/22: load using COCO API
        for image_link in image_links:
            images.append(self.load_image(image_link, coco_instance, local))

        #turn images into np array of 32-bit floats
        images = np.array(images, np.float32)
        return images



class CaptionData(object):
    def __init__(self, sentence, memory, output, score):
       self.sentence = sentence
       self.memory = memory
       self.output = output
       self.score = score

    def __cmp__(self, other):
        assert isinstance(other, CaptionData)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, CaptionData)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, CaptionData)
        return self.score == other.score



class TopN(object):
    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        self._data = []
