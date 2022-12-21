import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import pickle
import copy
import json

#tqdm is a library in Python which is used for creating Progress Meters or Progress Bars
from tqdm import tqdm

from utils.nn import NN
from utils.coco.coco import COCO as myCOCO

from pycocotools.coco import COCO as theyCOCO

from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import ImageLoader, CaptionData, TopN

LOCAL = True

class BaseModel(object):
    #when constructing object instance, need to pass the config instance, and the name of the annotations JSON file to be used for collecting images and captions
    def __init__(self, config, ann_file):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn

        #ADDED BY NWEINER 12/14/22: save the annotation file
        self.ann_file = ann_file

        print("model constructor: ann_file is", self.ann_file)

        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')

        self.image_shape = [224, 224, 3]
        self.nn = NN(config)

        #define global_step, a TensorFlow variable
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.build()

    def build(self):
        raise NotImplementedError()

    def train(self, sess, train_data):
        """ Train the model using the COCO train2014 data. """
        print("Training the model via train() in base_model.py...")

        #get the configuration details
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)

        train_writer = tf.summary.FileWriter(config.summary_dir, sess.graph)

        #ADDED BY NWEINER 12/14/22
        #instantiate an instance of COCO
        coco_caps = myCOCO(self.ann_file, config.num_train_data)

        #iterate over all training epochs
        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            #iterate over all training data batches
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                #get next batch of training data
                batch = train_data.next_batch()

                #from batch, extract the lists of image links, the corresponding captions, and the corresponding masks
                image_ids, image_links, sentences, masks = batch

                #load all images for this batch via COCO api and store them in a list
                #pass local as var to indicate whether to load image from COCO api url, or from local folder
                images = self.image_loader.load_images(image_links, config.local)
                
                feed_dict = {self.images: images, self.sentences: sentences, self.masks: masks}

                #run training session on this data batch
                _, summary, global_step = sess.run([self.opt_op, self.summary, self.global_step], feed_dict=feed_dict)


                if (global_step + 1) % config.save_period == 0:
                    self.save()

                train_writer.add_summary(summary, global_step)
            train_data.reset()

        #save the model
        self.save()

        train_writer.close()

        print("Training complete.")


    #a portion of the COCO data is used as evaluation data after the network is trained (it's taken from COCO "val" dataset)
    def eval(self, sess, eval_gt_coco, eval_data, vocabulary):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")

        config = self.config

        #results will be a list of dicts that contain the ID of the image being passe thru the trained network, and the corresponding caption generated
        results = []


        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        # Generate the captions for the images
        idx = 0

        #for num of batches
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
            #get next batch of eval data
            batch = eval_data.next_batch()

            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if k < eval_data.num_batches - 1 else eval_data.fake_count

            for l in range(eval_data.batch_size - fake_cnt):
                #get caption data that was generated for this image, as indices of words
                word_idxs = caption_data[l][0].sentence

                #get score
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)

                #append to results: results will be a list of dicts
                results.append({'image_id': int(eval_data.image_ids[idx]), 'caption': caption})

                #print("appending img id {} and caption {} to results".format(eval_data.image_ids[idx], caption))

                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = plt.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir, image_name + '_result.jpg'))


        fp = open(config.eval_result_file, 'w')
        #fp = open(config.eval_result_file, "w", encoding="utf8")

        json.dump(results, fp)

        #print(results)

        # Serializing json
        #json_object = json.dumps(results)
        
        #fp.write(json_object)

        fp.close()

        #evaluate these generated captions compared to the ground truths, using the scoring techniques (BLEU, etc.) in eval.py
        eval_result_coco = eval_gt_coco.loadRes(config) #the return here is a COCO object

        #instantiate a COCOEvalCap obj to do scoring
        scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)

        #run scoring
        scorer.evaluate()

        print("Evaluation complete.")


    #network can be tested on our own JPG images in the test/ folder
    def test(self, sess, test_data, vocabulary):
        """ Test the model using any given images. """
        print("Testing the model (test fxn in base_model.py)...")
        config = self.config

        if not os.path.exists(config.test_result_dir):
            os.mkdir(config.test_result_dir)

        captions = []
        scores = []


        # Generate the captions for the images
        for k in tqdm(list(range(test_data.num_batches)), desc='path'):
            batch = test_data.next_batch()

            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if k < test_data.num_batches-1 else test_data.fake_count


            for l in range(test_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                captions.append(caption)
                scores.append(score)

                # Save the result in an image file
                image_file = batch[l]
                image_name = image_file.split(os.sep)[-1]
                image_name = os.path.splitext(image_name)[0]
                img = plt.imread(image_file)

                plt.imshow(img)
                plt.axis('off')
                plt.title(caption)
                plt.savefig(os.path.join(config.test_result_dir, image_name + '_result.jpg'))


        # Save the generated captions to a CSV file in test folder
        results = pd.DataFrame({'image_files': test_data.image_links, 'caption': captions, 'prob': scores})
                                
        results.to_csv(config.test_result_file)


        print("Testing complete. Results are in {}".format(config.test_result_dir))



    def beam_search(self, sess, image_files, vocabulary):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states
        config = self.config

        images = self.image_loader.load_images(image_files, config.local)

        contexts, initial_memory, initial_output = sess.run(
            [self.conv_feats, self.initial_memory, self.initial_output],
            feed_dict = {self.images: images})

        partial_caption_data = []
        complete_caption_data = []

        for k in range(config.batch_size):
            initial_beam = CaptionData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       score = 1.0)
            partial_caption_data.append(TopN(config.beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(config.beam_size))

        # Run beam search
        for idx in range(config.max_caption_length):
            partial_caption_data_lists = []
            for k in range(config.batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else config.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((config.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32)

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32)

                memory, output, scores = sess.run(
                    [self.memory, self.output, self.probs],
                    feed_dict = {self.contexts: contexts,
                                 self.last_word: last_word,
                                 self.last_memory: last_memory,
                                 self.last_output: last_output})

                # Find the beam_size most probable next words
                for k in range(config.batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[0:config.beam_size+1]

                    # Append each of these words to the current partial caption
                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        beam = CaptionData(sentence,
                                           memory[k],
                                           output[k],
                                           score)
                        if vocabulary.words[w] == '.':
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

        results = []

        for k in range(config.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        return results


    def save(self):
        """ Save the model. """
        config = self.config

        #save the model parameters
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path + ".npy")))
        np.save(save_path, data)

        pickle_filename = os.path.join(config.save_dir, "config.pickle")

        #save the config info
        print("Saving the config info into %s..." % pickle_filename)
        info_file = open(pickle_filename, "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)


        info_file.close()

        print("Model saved.")



    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir, str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path, allow_pickle = True, encoding = 'latin1').item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)



    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path, allow_pickle = True, encoding = 'latin1').item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse = True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)
