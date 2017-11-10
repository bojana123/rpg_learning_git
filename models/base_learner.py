import os
import sys
import time
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset
import keras.backend as K
from keras.utils.generic_utils import Progbar
from unipath import Path
##########################################################
# IMPORT YOUR FAVORITE NETWORK HERE (in place of resnet8)#
##########################################################
from .nets import resnet8_tf as prediction_network
import matplotlib.pyplot as plt

#############################################################################
# IMPORT HERE A LIBRARY TO PRODUCE ALL THE FILENAMES (and optionally labels)#
# OF YOUR DATASET. HAVE A LOOK AT `DirectoryIterator' FOR AN EXAMPLE        #
#############################################################################
sys.path.append("../")
from data import DirectoryIterator

class Learner(object):
    def __init__(self):
        pass

    def dataset_map(self, filename, label):
        """Consumes the inputs queue.
        Args:
            inputs_queue: A FIFO queue of filenames and labels
        Returns:
            Two tensors: the decoded images, and the labels.
        """
        #########################################################
        # CHANGE THIS ONLY IF YOU'RE _NOT_ DOING CLASSIFICATION #
        #########################################################
        label_seq = tf.cast(label, dtype=tf.float32)
        file_content = tf.read_file(filename)
        image_seq = tf.image.decode_png(file_content, channels=3)
        # Resize images to target size and preprocess them
        image_seq = tf.image.resize_images(image_seq,
                                [self.config.img_height, self.config.img_width])
        image_seq = self.preprocess_image(image_seq)
        return image_seq, label_seq

    def preprocess_image(self, image):
        #############################
        # DO YOUR PREPROCESSING HERE#
        #############################
        """ Preprocess an input image.
        Args:
            Image: A uint8 tensor
        Returns:
            image: A preprocessed float32 tensor.
        """
        print( 'HOLA' )
        print(self.config.crop_img_width)
        print( 'ADIOS' )
        
         # Cropping
        if self.config.crop_img_width:
                top_left_y = 0
                top_left_x = tf.shape(image)[1]/2 - self.config.crop_img_width/2
                top_left_x = tf.cast(top_left_x, tf.int32)
                image = tf.image.crop_to_bounding_box(image, top_left_y, top_left_x,
                                                      self.config.crop_img_height,
                                                      self.config.crop_img_width)
        
        if self.config.frame_mode == 'dvs':
            event_img = tf.split(image, 3, axis=-1)
            print(np.shape(image))
 
        # Normalize positive-event image
            pos_events = tf.cast(event_img[0], tf.float32)
            print(np.shape(event_img[0]))
            norm_pos_events = tf.divide(pos_events, self.perc_dict['pos_sup'])
            norm_pos_events = tf.squeeze(norm_pos_events, axis=-1)

        # Normalize negative-event image
            neg_events = tf.cast(event_img[-1], tf.float32)
            norm_neg_events = tf.divide(neg_events, self.perc_dict['neg_sup'])
            norm_neg_events = tf.squeeze(norm_neg_events, axis=-1)
            
            input_img = tf.stack([norm_pos_events, norm_neg_events], axis=-1)
            
        elif self.config.frame_mode == 'aps':
            if self.config.img_width:
                input_img = tf.image.resize_images(image,[self.config.img_height, self.config.img_width])
            input_img = tf.divide(input_img, 255)
            input_img = tf.squeeze(input_img, axis=-1)
            input_img = tf.expand_dims(input_img, axis=-1)
            input_img = tf.cast(input_img, dtype=tf.float32)
            
        else:
            max_diff = tf.costant(np.log(255+ 1e-3) - np.log(0 + 1e-3), dtype=tf.float32)
            if self.comfig.img.width:
                input_img = tf.image.resize_images(image, [self.config.img_height, self.config.img_width])
                
            slices = tf.split(image, 3, axis=-1)
            prev_img = tf.add(tf.cast(slices[0], tf.float32), 1e-3)
            prev_img = tf.squeeze(prev_img, axis=-1)
            
            post_img = tf.add(tf.cast(slices[-1], tf.float32), 1e-3)
            post_img = tf.squeeze(prev_img, axis=-1)
            
            input_img = tf.divide(tf.subtract(tf.log(post_img), tf.log(prev_img)), max_diff)
            input_img = tf.expand_dims(input_img, axis=-1)
            
        return input_img

        #image = tf.cast(image, dtype=tf.float32)
        #image = tf.divide(image, 255.0)
        #return image

    def get_filenames_list(self, directory):
        """ This function should return all the filenames of the
            files you want to train on.
            In case of classification, it should also return labels.

            Args:
                directory: dataset directory
            Returns:
                List of filenames, [List of associated labels]
        """
        iterator = DirectoryIterator(directory, shuffle=False)
        print("Shape of ground truth: ", np.shape(iterator.ground_truth))
        print("groud_truth[0]: ", iterator.ground_truth[0])
        return iterator.filenames, iterator.ground_truth

    def build_train_graph(self):
        with tf.name_scope("data_loading"):
            #########################################################
            # In case of classification, this should be unchanged.  #
            # Otherwise, adapt to load your inputs                  #
            #########################################################
        #
            #seq_len = self.config.seq_len
            #file_list, labels_list = self.get_filenames_list(
             #                                   self.config.train_dir,
              #                                  'train')
            
            # Load percentiles for positive and negative event normalization and
            # set number of channels
            self.channels = 3 
            # Dictionary of percentiles to normalize DVS input data
            self.perc_dict = {}
            # Superior percentile for positive events
            self.perc_dict['pos_sup'] = 17.0
            # Superior percentile for negative events
            self.perc_dict['neg_sup'] = 17.0

                    
            seed = random.randint(0, 2**31 - 1)

            # Load the list of training files into queues
            file_list, labels_list = self.get_filenames_list(
                                                self.config.train_dir)
                                                
            # Build the dataset
            dataset = Dataset.from_tensor_slices((file_list, labels_list))
            dataset = dataset.map(self.dataset_map,
                                num_threads=self.config.num_threads,
                                output_buffer_size = self.config.capacity_queue)
            dataset = dataset.shuffle(buffer_size=1000, seed=seed)
            dataset = dataset.batch(self.config.batch_size)
            dataset = dataset.repeat(self.config.max_epochs)
            iterator = dataset.make_initializable_iterator()
            image_batch, label_batch = iterator.get_next()
            print("shape of label_batch: ", label_batch.get_shape())

        with tf.name_scope("CNN_prediction"):
            is_training_ph = tf.placeholder(tf.bool, shape=(),
                                            name="is_training")
            logits = prediction_network(image_batch,
                                        l2_reg_scale=self.config.l2_reg_scale,
                                        is_training=is_training_ph,
                                        output_dim=self.config.output_dim)
            print("shape of logits: ", logits.get_shape())

        with tf.name_scope("compute_loss"):
            ####################################
            # CHANGE HERE TO YOUR PROBLEM LOSS #
            ####################################
            #train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #    labels=label_batch, logits=logits)
            #train_loss = tf.reduce_mean(train_loss)
            train_loss = tf.reduce_mean(tf.square(logits-label_batch))
            #train_loss = tf.metrics.mean_squared_error(label_batch, logits)

            print("shape of train_loss: ", train_loss.get_shape())

     #   with tf.name_scope("accuracy"):
      #      pred_out = tf.cast(tf.argmax(logits, 1), tf.int32)
       #     correct_prediction = tf.equal(label_batch, pred_out)
         #   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope("train_op"):
            #######################################################
            # LEAVE UNCHANGED (Adam optimizer is usually the best)#
            #######################################################
            reg_losses = tf.reduce_sum(
                          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            train_vars = [var for var in tf.trainable_variables()]
            optimizer  = tf.train.AdamOptimizer(self.config.learning_rate,
                                             self.config.beta1)
            self.grads_and_vars = optimizer.compute_gradients(train_loss + reg_losses,
                                                          var_list=train_vars)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars)
            self.global_step = tf.Variable(0,
                                           name='global_step',
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step,
                                              self.global_step+1)

        #######################################################################
        # ADD HERE ALL THE TENSORS YOU WANT TO RUN OR SUMMARIZE IN TENSORBOARD#
        #######################################################################

        self.logits = logits
        self.labels = label_batch
        #self.accuracy = accuracy
        self.is_training = is_training_ph
        self.iterator_op = iterator.initializer
        self.steps_per_epoch = \
            int(math.ceil(len(file_list)/self.config.batch_size))
        self.total_loss = train_loss

    def collect_summaries(self):
        """Collects all summaries to be shown in the tensorboard"""
        #######################################################
        # ADD HERE THE VARIABLES YOU WANT TO SEE IN THE BOARD #
        #######################################################
        tf.summary.scalar("train_loss", self.total_loss)
        #tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.histogram("logits_distribution", self.logits)
        #tf.summary.histogram("predicted_out_distributions", tf.argmax(
                            #self.logits,1))
        tf.summary.histogram("ground_truth_distribution", self.labels)
        ###################################################
        # LEAVE UNCHANGED (gradients and tensors summary) #
        ###################################################
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)
        for grad, var in self.grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to {}/model-{}".format(checkpoint_dir,
                                                             step))
        if step == 'latest':
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def train(self, config):
        """High level train function.
        Args:
            config: Configuration dictionary
        Returns:
            None
        TODO: Add progbar from keras
        """
        self.config = config
        self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                        for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in \
            tf.trainable_variables()] +  [self.global_step], max_to_keep=5)
        sv = tf.train.Supervisor(logdir=config.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        with sv.managed_session() as sess:
            print("Number of trainable params: {}".format(sess.run(parameter_count)))
            if config.resume_train:
                print("Resume training from previous checkpoint")
                checkpoint = tf.train.latest_checkpoint(
                                                config.checkpoint_dir)
                self.saver.restore(sess, checkpoint)

            progbar = Progbar(target=self.steps_per_epoch)
            sess.run(self.iterator_op)
            for step in range(1, config.max_steps):
                if sv.should_stop():
                    break
                start_time = time.time()
                fetches = { "train" : self.train_op,
                              "global_step" : self.global_step,
                              "incr_global_step": self.incr_global_step
                             }
                if step % config.summary_freq == 0:
                    #########################################################
                    # ADD HERE THE TENSORS YOU WANT TO EVALUATE (maybe loss)#
                    #########################################################
                    fetches["loss"] = self.total_loss
                    #fetches["accuracy"] = self.accuracy
                    fetches["summary"] = sv.summary_op

                # Runs the series of operations
                ######################################################
                # REMOVE THE LEARNING PHASE IF NOT USING KERAS MODELS#
                ######################################################
                try:
                    results = sess.run(fetches,
                                       feed_dict={ self.is_training : True })
                    progbar.update(step % self.steps_per_epoch)
                except tf.errors.OutOfRangeError:
                    print("-------------------------------")
                    print("Training completed successfully")
                    print("-------------------------------")
                    break # Max epochs reached

                gs = results["global_step"]
                if step % config.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil( gs /self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f "
                         \
                       % (train_epoch, train_step, self.steps_per_epoch, \
                                time.time() - start_time, results["loss"]))
                          #results["accuracy"]))
                            

                if step % config.save_latest_freq == 0:
                    self.save(sess, config.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, config.checkpoint_dir, gs)
                    progbar = Progbar(target=self.steps_per_epoch)

    def build_test_graph(self):
        """This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or some other utilities.
           Here, data will be passed though placeholders and not via
           input queues.
        """
        ##################################################################
        # UNCHANGED FOR CLASSIFICATION. ADAPT THE INPUT TO OTHER PROBLEMS#
        ##################################################################
        image_height, image_width = self.config.img_height, \
                                    self.config.img_width
        input_uint8 = tf.placeholder(tf.uint8, [None, image_height,
                                    image_width, 1], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)

        gt_labels = tf.placeholder(tf.uint8, [None], name='gt_labels')
        input_labels = tf.cast(gt_labels, tf.int32)

        ################################################
        # DONT CHANGE NAMESCOPE (NECESSARY FOR LOADING)#
        ################################################
        with tf.name_scope("CNN_prediction"):
            logits = prediction_network(input_mc,
                    l2_reg_scale=self.config.l2_reg_scale, is_training=False,
                    output_dim=self.config.output_dim)

        ###########################################
        # ADAPT TO YOUR LOSSES OR TESTING METRICS #
        ###########################################

        with tf.name_scope("compute_loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=input_labels, logits= logits)
            loss = tf.reduce_mean(loss)

        with tf.name_scope("accuracy"):
            pred_out = tf.cast(tf.argmax(logits, 1), tf.int32)
            correct_prediction = tf.equal(input_labels, pred_out)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ################################################################
        # PUT HERE THE PLACEHOLDERS YOU NEED TO USE, AND OPERATIONS YOU#
        # WANT TO EVALUATE                                             #
        ################################################################
        self.inputs = input_uint8
        self.gt_labels = gt_labels
        self.total_loss = loss
        self.predictions = pred_out
        self.accuracy = accuracy

    def setup_inference(self, config):
        """Sets up the inference graph.
        Args:
            config: config dictionary.
        """
        self.config = config
        self.build_test_graph()

    def inference(self, inputs, sess):
        """Outputs a dictionary with the results of the required operations.
        Args:
            inputs: Dictionary with variable to be feed to placeholders
            sess: current session
        Returns:
            results: dictionary with output of testing operations.
        """
        ################################################################
        # CHANGE INPUTS TO THE PLACEHOLDER YOU NEED, AND OUTPUTS TO THE#
        # RESULTS OF YOUR OPERATIONS                                   #
        ################################################################
        results = {}
        results['loss'], results['accuracy'] = sess.run([self.total_loss,
                self.accuracy], feed_dict= {self.inputs: inputs['images'],
                                            self.gt_labels: inputs['labels']})
        return results
