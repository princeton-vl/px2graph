import tensorflow as tf
import numpy as np

class Loader:

    def __init__(self, opt, task, ds, train_flag):
        self.opt = opt
        self.ds = ds
        self.num_threads = opt.num_data_threads

        # Index queue
        self.input_idxs = tf.placeholder(tf.int64, shape=[None, 2])
        idx_queue = tf.FIFOQueue(1e8, tf.int64)
        self.enq_idxs = idx_queue.enqueue_many(self.input_idxs)
        get_idx = idx_queue.dequeue()

        # Image loading queue
        img_queue = tf.FIFOQueue(opt.max_queue_size, task.proc_arg_dtype)
        load_data = tf.py_func(task.load_sample_data, [get_idx], task.proc_arg_dtype)
        enq_img = img_queue.enqueue(load_data)
        init_sample = img_queue.dequeue()

        # Preprocessing queue
        # (for any preprocessing that can be done with TF operations)
        data_queue = tf.FIFOQueue(opt.max_queue_size, task.data_arg_dtype,
                                  shapes=task.data_shape)
        enq_data = data_queue.enqueue(task.preprocess(init_sample, train_flag))
        self.get_sample = data_queue.dequeue_many(opt.batchsize)

        # Queue runners
        self.img_runner = tf.train.QueueRunner(img_queue, [enq_img] * opt.num_data_threads)
        self.data_runner = tf.train.QueueRunner(data_queue, [enq_data] * opt.num_data_threads)


    def start_threads(self, sess):
        # Start queueing threads
        self.coord = tf.train.Coordinator()
        self.img_threads = self.img_runner.create_threads(sess, coord=self.coord, daemon=True, start=True)
        self.data_threads = self.data_runner.create_threads(sess, coord=self.coord, daemon=True, start=True)


    def start_epoch(self, sess, split, train_flag, num_samples=-1, flag_val=None, in_order=False):
        if flag_val is None: flag_val = split == 'train'
        idx_ref = self.opt.idx_ref[split]

        # Choose indices to load (randomly shuffled if in_order is false)
        print("Loading from %s, %d samples available" % (split, idx_ref.shape[0]))
        idxs = np.arange(idx_ref.shape[0])
        if idxs.shape[0] < num_samples: idxs = idxs.repeat(np.ceil(num_samples / idxs.shape[0]), 0)
        if not in_order: np.random.shuffle(idxs)
        if num_samples != -1: idxs = idxs[:num_samples]
        idxs = idx_ref[idxs]

        idxs = np.stack((idxs, np.array([int(flag_val)]*idxs.shape[0])), 1)
        sess.run(self.enq_idxs, feed_dict={self.input_idxs:idxs, train_flag:flag_val})

        return idxs
