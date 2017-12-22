import tensorflow as tf
import numpy as np

from px2graph.models import layers

#cnv = layers.res_block
cnv = lambda x,y: layers.cnv(x,y,3)
pool = layers.max_pool

def hourglass(inp, f, depth, max_depth, scale_feats=None):

    # Skip branch
    skip_branch = cnv(inp, f)

    # Lower branch
    with tf.variable_scope('res_%d' % (2**(depth+1))):

        tmp_f = f
        if scale_feats is not None: tmp_f *= scale_feats[max_depth - depth]

        c1 = cnv(pool(inp, 2, 2), tmp_f)

        if depth > 1: c2 = hourglass(c1, tmp_f, depth - 1,
                                     max_depth, scale_feats)
        else: c2 = cnv(c1, tmp_f)

        c3 = cnv(c2, f)
        upsampled = tf.image.resize_images(c3, inp.shape[1:3])

    return skip_branch + upsampled


def initialize(inp, opt, train_flag, suffix='', reuse=None):

    with tf.variable_scope('hourglass'+suffix, reuse=reuse):
        return hourglass(inp, opt.num_feats, 4, 4, opt.scale_feats)
