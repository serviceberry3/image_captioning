#!/usr/bin/python
import tensorflow.compat.v1 as tf #this will be Tf 2

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

#flags passed to the application

#CHANGED by nweiner
#FLAGS = tf.app.flags.FLAGS
FLAGS = tf.flags.FLAGS


#define all of the flag options

#CHANGED by nweiner from tf.flags to tf.compat.v1.flags

#this cmd registers a flag whose value can be any string.
tf.flags.DEFINE_string('phase', 
'train', #the second entry here gives the default value for the flag
                       'The phase can be #train, #eval or #test, where #train is the default') #the third entry gives the help description for the flag

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If specified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', True,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')




#main program entrypoint
def main(argv):
    print("Calling main() in main.py...")

    #instantiate a Config object, and set some of its parameters based on user-passed args
    #some parameters are also set by default in Config's constructor
    config = Config()

    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size

    tf.disable_eager_execution()


    #create a TensorFlow session
    with tf.Session() as sess:
        if FLAGS.phase == 'train':
            #if user specified training, then train the network
            print("\nUser specified train, so training network...\n")

            #prepare the training data
            data = prepare_train_data(config)

            #build the model
            model = CaptionGenerator(config, "./train/captions_train2014.json")

            #run the TensorFlow training session
            sess.run(tf.global_variables_initializer())

            #if user specified checkpoint to load model weights from, load it
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)

            #if user specified that CNN portion of model should be loaded from pretrained CNN file, load it
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            
            #get_default_graph() returns the default graph for the current thread
            tf.get_default_graph().finalize()

            #run train() method in base_model
            model.train(sess, data)
            
        
        #if user specified to do eval
        elif FLAGS.phase == 'eval':
            #if user specified eval, then do evaluation phase
            print("User specified eval, so doing evaluation...")

            coco, data, vocabulary = prepare_eval_data(config)

            model = CaptionGenerator(config, "./val/captions_val2014.json")
            model.load(sess, FLAGS.model_file)

            tf.get_default_graph().finalize()
            model.eval(sess, coco, data, vocabulary)


        #if user specified to test network
        elif FLAGS.phase == 'test':
            #otherwise, do testing phase
            print("User specified test, so testing trained network...")

            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)

        else:
            #otherwise, phase arg is invalid
            print('ERROR: the phase can be #train, #eval or #test, where #train is the default')

            return



#MAIN PROGRAM ENTRY PT
if __name__ == '__main__':
    print("Calling tf app.run()...")

    #run the TensorFlow app
    tf.app.run()
