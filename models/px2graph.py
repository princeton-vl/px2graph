import tensorflow as tf
import numpy as np

from px2graph.models import layers, hg

def initialize(inp, opt, train_flag):
    """ Initialize pixels to graph network architecture.

    This is the full pipeline to output a set of object and relationship
    detections. Parsing the network output is defined in px2graph.task.graph.

    Args:
        inp: List of all network inputs which includes: the input image; box
            detections provided as additional input in easier task settings; a
            set of (x,y) coordinates defining where the ground truth nodes and
            edges are (not used at test time)

            0  : Input image        (in_res x in_res x 3)
            1  : Input heatmap      (out_res x out_res x ?)
            2  : Node keypoint ref  (2 x n x 2)

        opt: Namespace holding all command line options.
        train_flag: Tensorflow boolean indicating whether or not a sample is
            a training or test sample.

    Returns:
        A list of network outputs as follows (n = max number of nodes, hard
        coded to avoid excessive dynamic computation):

            0  : Heatmaps           (out_res x out_res x num_ch)
            1  : Obj hm activations (2*n)
            2  : Obj scores         (2*n x obj_slots x 1)
            3  : Obj class scores   (2*n x obj_slots x num_obj_cats)
            4  : Obj tags           (2*n x obj_slots x tag_dim)
            5  : Bbox part 1        (2*n x obj_slots x bbox_channels)
            6  : Bbox part 2        (2*n x obj_slots x bbox_channels)
            7  : Rel hm activations (2*n)
            8  : Rel scores         (2*n x rel_slots x 1)
            9  : Rel classes        (2*n x rel_slots x num_rel_cats)
            10 : Subject tags       (2*n x rel_slots x tag_dim)
            11 : Object tags        (2*n x rel_slots x tag_dim)
            12 : Keypoint ref       (2 x 2*n x 2)
    
    """

    # Unpack inputs
    inp_img, inp_hm, kp_ref = inp
    batchsize = tf.shape(inp_img)[0]

    ### ======================================================================
    ### Main network body
    ### ======================================================================

    f = opt.num_feats
    cnv = lambda x,y: layers.cnv(layers.relu(x),y,3)
    pool = layers.max_pool

    with tf.variable_scope('network'):
        with tf.variable_scope('preprocess'):
            c1 = layers.cnv(inp_img, 64, 7, strides=2)      # 512 -> 256
            p1 = pool(c1, 3, 2)                             # 256 -> 128
            c2 = cnv(p1, f)
            c3 = cnv(c2, f)
            c4 = cnv(c3, f)
            p2 = pool(c4, 2, 2)                             # 128 -> 64
            c5 = cnv(p2, f)
            curr_feats = cnv(c5, f)

        # Incorporate prior detections (either ground truth or from RPN)
        if opt.sg_task != 'SG' or opt.use_rpn:
            print("Using input detections")
            with tf.variable_scope('input_detections'):
                d1 = cnv(inp_hm, f)
                d2 = cnv(d1, f)
                curr_feats += d2

        intermediate_feats = []
        heatmap_preds = []

        # Stacked hourglass
        for i in range(opt.num_stack):

            hg_out = hg.initialize(curr_feats, opt, train_flag, suffix='_%d'%i)

            f1 = cnv(hg_out, f)
            f2 = layers.cnv(f1, f, 1)

            intermediate_feats += [f2]
            heatmap_preds += [layers.cnv(layers.relu(f1), opt.num_hm_channels, 1)]

            if i < opt.num_stack - 1: curr_feats += cnv(f1, f)

        hg_feats = tf.reduce_mean(intermediate_feats,0)
        heatmaps = tf.reduce_mean(heatmap_preds,0)

    ### ======================================================================
    ### Extract features and predict graph elements
    ### ======================================================================

    output_categories = ['objects', 'relationships']
    output_ref = [[1, opt.num_cats, opt.tag_dim, opt.bbox_channels, opt.bbox_channels],
                  [1, opt.num_rels, opt.tag_dim, opt.tag_dim]]
    num_slot_ref = [opt.obj_slots, opt.rel_slots]
    fc_out_size = [output_ref[i] * num_slot_ref[i] for i in range(len(output_ref))]
    outputs = {}

    # Get predicted keypoints based on heatmap activations
    pred_obj_kps, _ = layers.top_k(heatmaps[:,:,:,0], opt.max_nodes, do_nms=opt.obj_hm_nms)
    pred_rel_kps, _ = layers.top_k(heatmaps[:,:,:,1], opt.max_nodes, do_nms=opt.rel_hm_nms)

    # Ground truth keypoints are provided as (x,y), but when we do indexing we need (y,x)
    gt_keypoints = tf.to_int32(tf.reverse(kp_ref, [-1]))
    pred_keypoints = tf.stack([pred_obj_kps, pred_rel_kps], 1)
    all_keypoints = tf.concat([gt_keypoints, pred_keypoints], 2)

    def get_preds(ref_idx):
        # Collect heatmap activations and features from gt and predicted keypoints
        tmp_out = [layers.gather(heatmaps[:,:,:,ref_idx], all_keypoints[:,ref_idx])]
        tmp_feats = tf.reshape(layers.gather(hg_feats, all_keypoints[:,ref_idx]), [-1, f])

        # For each output, pass features through a couple of fully connected layers
        for out_idx, out_size in enumerate(fc_out_size[ref_idx]):
            with tf.variable_scope('out_%d' % out_idx):
                f1 = layers.dense(tmp_feats, f)
                f2 = layers.dense(layers.relu(f1), out_size)
                tmp_out += [tf.reshape(f2, [batchsize, -1, out_size])]

        return tmp_out

    for i,out_cat in enumerate(output_categories):
        outputs[out_cat] = []

        with tf.variable_scope(out_cat):
            num_outs = len(output_ref[i])
            tmp_preds = get_preds(i)
            outputs[out_cat] += [tmp_preds[0]]

            # Combine predictions across multiple "slots"
            tmp_preds = tmp_preds[1:]
            tmp_idx_ref = np.arange(num_outs * num_slot_ref[i]).reshape([-1, num_outs]).T
            for j in range(num_outs):
                tmp_outs = [tmp_preds[tmp_idx_ref[j][k]] for k in range(num_slot_ref[i])]
                outputs[out_cat] += [tf.stack(tmp_outs, axis=2)]

    return [
        tf.sigmoid(heatmaps),
        *outputs['objects'],
        *outputs['relationships'],
        all_keypoints
    ]
