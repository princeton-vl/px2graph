import numpy as np
import tensorflow as tf
import pickle

class Task:
    """ Base set of functions for all tasks.
    """

    # Define the type/shape of all input/label tensors
    proc_arg_dtype = []
    data_arg_dtype = []
    np_dtype = [tmp.as_numpy_dtype for tmp in proc_arg_dtype]
    data_shape = ()
    num_inputs = 1      # Where to split between inputs/labels
    data_labels = []    # Provide a reference to access tensors
    
    def __init__(self, opt, ds):
        self.opt = opt
        self.ds = ds

    def load_sample_data(self, idx):
        idx, train_flag = idx
        return

    def get_data_shape(self):
        return self.data_shape_ref

    def preprocess(self, data, train_flag):
        return data

    def loss(self, net, label):
        # Calculate network loss
        self.loss_ref = ['default_loss']
        self.__aux_data = [tf.zeros([])]
        return [tf.zeros([])]

    def postprocess(self, idx, net, label):
        # Postprocessing to calculate accuracy and produce final predictions
        pred = {'default_pred':tf.zeros([])}
        acc = {'default_acc':tf.zeros([])}
        return pred, acc

    def setup_summaries(self, net, inp, label, loss, pred, accuracy):
        # Set up logging of accuracy and loss, as well as image summaries
        summaries, image_summaries = {}, {}
        for s in ['train', 'valid']:
            tmp_scalar_summaries = []

            for i,k in enumerate(self.loss_ref):
                tmp_scalar_summaries += [tf.summary.scalar('%s_%s_loss' % (s,k), loss[i])]
            for k in accuracy:
                tmp_scalar_summaries += [tf.summary.scalar('%s_%s'%(s,k), accuracy[k])]

            # Assumes first input is an image
            tmp_image_summaries = [tf.summary.image(s + '_input', inp[0])]

            summaries[s] = tf.summary.merge(tmp_scalar_summaries)
            image_summaries[s] = tf.summary.merge(tmp_image_summaries)

        return summaries, image_summaries
