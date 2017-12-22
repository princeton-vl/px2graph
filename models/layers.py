import tensorflow as tf

# Wrap tensorflow calls so default padding is 'same'

def avg_pool(*args, padding='same', **kargs):
    return tf.layers.average_pooling2d(*args, **kargs, padding=padding)
def max_pool(*args, padding='same', **kargs):
    return tf.layers.max_pooling2d(*args, **kargs, padding=padding)
def cnv(*args, padding='same', **kargs):
    return tf.layers.conv2d(*args, **kargs, padding=padding,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
def dense(*args, **kargs):
    return tf.layers.dense(*args, **kargs,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())

relu = tf.nn.relu

def gated(x):
    x0, x1 = tf.split(x, num_or_size_splits=2, axis=-1)
    return tf.nn.sigmoid(x0) * tf.nn.tanh(x1)

def res_block(inp, f, k=3, activation=relu):
    f_ = f // 2 # Bottleneck size

    inp_ = activation(inp)
    c1 = activation(cnv(inp_, f_, 1))
    c2 = activation(cnv(c1, f_, k))
    c3 = cnv(c2, f, 1)

    # Check whether input channel dimension matches f
    if int(inp.shape[-1]) != f:
        inp = cnv(inp, f, 1)

    return c3 + inp

def gather(arrs, idxs):
    # Batched gather_nd
    batchsize = int(arrs.shape[0])
    tmp_vals = []
    for i in range(batchsize):
        tmp_vals += [tf.gather_nd(arrs[i], idxs[i])]
    tmp_vals = tf.stack(tmp_vals, 0)
    return tmp_vals

def top_k(arr, k, batch=True, do_nms=0):
    # Return sorted set of keypoints and scores
    dims = [int(d) for d in arr.shape]
    last_idx = 0

    if do_nms:
        local_max = max_pool(tf.expand_dims(arr,3), 3, 1)
        local_max = tf.to_float(tf.equal(arr, local_max[:,:,:,0]))
        arr = local_max * arr

    if batch:
        # Assumes first dimension is batch dimension
        sorted_scores, sorted_idxs = tf.nn.top_k(tf.reshape(arr, [dims[0], -1]), k=k)
    else:
        sorted_scores, sorted_idxs = tf.nn.top_k(tf.reshape(arr, [-1]), k=k)
        last_idx -= 1

    # Unravel indices
    idxs = []
    tmp_val = sorted_idxs
    for i in range(len(dims)-1,last_idx,-1):
        idxs = [tmp_val % dims[i]] + idxs
        tmp_val = tmp_val // dims[i]
    sorted_kps = tf.stack(idxs, 2 + last_idx)

    return sorted_kps, sorted_scores
