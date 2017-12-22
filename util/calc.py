""" Miscellaneous helper functions for calculations. """

import tensorflow as tf
import numpy as np
import os

curr_dir = os.path.dirname(__file__)
munkres = tf.load_op_library(curr_dir + '/hungarian.so')

def py_max_match(scores):
    # Backup if you have trouble getting munkres-tensorflow compiled (much slower)
    from munkres import Munkres
    m = Munkres()
    tmp = m.compute(-scores)
    return np.array(tmp).astype(np.int32)

def match_and_shuffle(pred, gt, to_shuffle, split_size_ref):    
    # Get max matchup
    vec_comparison = -tf.matmul(gt, pred, transpose_b=True)
    vec_comparison = tf.expand_dims(vec_comparison, 0)
    matchup = munkres.hungarian(vec_comparison)[0]

    # Shuffle vectors
    result = tf.gather(to_shuffle, matchup)
    split_idxs = np.cumsum([0]+split_size_ref)[:-1]
    return [tf.slice(result, [0, split_idxs[idx]], [-1, split_size_ref[idx]])
              for idx in range(len(split_size_ref))]

def top_k_threshold(scores, k, thr):
    num_vals = tf.size(tf.where(scores > thr))
    return tf.nn.top_k(scores, k=tf.minimum(k, num_vals))

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # --------------------------------------------------------
    # (from) Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# =============================================================================
# Loss functions
# =============================================================================

def mse_loss(pred, gt):
    return tf.reduce_mean(tf.square(pred - gt))

def bce_loss(pred, gt):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=gt))

def activation_loss(acts, n):
    # First n values of acts should be 1, everything else 0
    ref_arr = tf.pad(tf.ones([n]), [[0, tf.shape(acts)[0] - n]])
    return bce_loss(acts, ref_arr)

def softmax_loss(pred, gt):
    tmp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
    return tf.reduce_mean(tmp_loss)

def hinge_loss(pred, gt):
    tmp_gt = 2 * tf.one_hot(gt, tf.shape(pred)[1]) - 1
    tmp_loss = tf.reduce_sum(tf.pow(tf.maximum(0., .5 - tmp_gt * pred),2),1)
    return tf.reduce_mean(tmp_loss)

def huber_loss(pred, gt, delta=1.0):
    # from stanford cs20si slides
    residual = tf.abs(pred - gt)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.reduce_mean(tf.where(condition, small_res, large_res))

def classify_loss(pred, gt):
    filter_idxs = tf.squeeze(tf.where(gt > -1), 1)
    pred = tf.gather(pred, filter_idxs)
    gt = tf.gather(gt, filter_idxs)
    return softmax_loss(pred, gt)

def push_apart(tags, tag_margin=8):
    # Get ref idxs
    n = tf.shape(tags)[0]
    m = [tf.reshape(m,[-1]) for m in tf.meshgrid(tf.range(n),tf.range(n))]
    idxs = tf.squeeze(tf.where(m[1] < m[0]),1)
    ref_idxs = tf.stack([tf.gather(m[1], idxs), tf.gather(m[0], idxs)],0)
    to_compare = tf.gather(tags, ref_idxs)

    # Calculate distances
    tag_dist = tf.pow(tf.reduce_sum(tf.pow(to_compare[0] - to_compare[1], 2), 1), .5)
    if tag_margin:
        tmp_cost = tf.pow(tf.maximum(tag_margin - tag_dist, 0), 2)
        return tf.reduce_mean(tmp_cost)
    else:
        exp_costs = tf.exp(-tf.pow(tag_dist, 2))
        cost_mask = tf.to_float(tf.greater(exp_costs, 1e-2))
        exp_costs = exp_costs * cost_mask
        return tf.reduce_mean(exp_costs)

def push_pull_loss(idx_ref, embed_pred, embed_ref, tag_margin=0):
    """ Calculate associative embedding loss

    Assume there are k groups whose embeddings we want to distinguish, we
    define a reference for each group in 'embed_ref' (k x tag_dim). In
    the context of multiperson pose this can be the mean embedding for each
    person, and for scene graphs this will be the embedding output for
    each object. Then, 'embed_pred' holds all of the predictions
    that refer to any of the k groups, 'idx_ref' will specify which group each
    item in 'embed_pred' is assigned to.

    Args:
        idx_ref: index reference (n x 1)
            (indices in idx_ref can range from 0..k-1)
        embed_pred: set of embedding predictions (n x dim)
        embed_ref: reference targets for each group (k x dim)
    
    Returns:
        loss: List where first penalty pulls together values belonging to same
            group, and second penalty pushes apart values across groups
    """

    # Get reference for each prediction
    embed_gt = tf.gather_nd(embed_ref, idx_ref)

    # Calculate pull loss
    pull_loss = mse_loss(embed_pred, embed_gt) # can also use a huber loss here
    # Apply push loss to mean vectors
    push_loss = tf.cond(tf.shape(embed_ref)[0] > 1,
                        lambda: push_apart(embed_ref, tag_margin),
                        lambda: tf.constant(0, tf.float32))

    return [pull_loss, push_loss]


# =============================================================================
# Bounding boxes
# =============================================================================

# Bounding boxes will usually be represented as a 2x2 array, defining the
# upper left and bottom right coordinates as [[x1,y1],[x2,y2]], some
# functions may use a flattened version of this

def flip_boxes(boxes, im_width):
    boxes = boxes.copy()
    boxes[:,:,0] = im_width - 1 - boxes[:,:,0]
    for i in range(boxes.shape[0]):
        x1, x2 = boxes[i,:,0]
        boxes[i,0,0] = x2
        boxes[i,1,0] = x1
    return boxes

def box_inbounds(box, img_res):
    ht, wd = img_res
    x1, y1 = box[0]
    x2, y2 = box[1]
    return not (x1 >= wd or y1 >= ht or x2 < 0 or y2 < 0)

def tf_decode_boxes(anchors, pred_kps, pred_idx, pred_reg, inp_res, out_res):
    # Given anchor box reference and predictions, decode network output to
    # define final bounding boxes. Remap from network out_res to inp res.

    num_anchors = len(anchors)
    anchor_idxs = tf.to_int32(tf.argmax(pred_idx[:,:num_anchors], 1))
    anchor_idxs = tf.stack([tf.range(tf.size(anchor_idxs)), anchor_idxs], 1)

    tmp_bbox_vals = tf.reshape(pred_reg, [-1, num_anchors, 4])
    tmp_bbox_vals = tf.gather_nd(tmp_bbox_vals, anchor_idxs)
    tx,ty,tw,th = tf.unstack(tmp_bbox_vals, axis=1)

    w_a_ref = (anchors[:,2] - anchors[:,0]).astype(np.float32)
    h_a_ref = (anchors[:,3] - anchors[:,1]).astype(np.float32)
    w_a = tf.gather(w_a_ref, anchor_idxs[:,1])
    h_a = tf.gather(h_a_ref, anchor_idxs[:,1])
    x_a = (tf.to_float(pred_kps[:,1]) + .5) * inp_res / out_res
    y_a = (tf.to_float(pred_kps[:,0]) + .5) * inp_res / out_res

    # Get box centers and widths/heights
    x_ = (tx * w_a) + x_a
    y_ = (ty * h_a) + y_a
    w_ = tf.exp(tw) * w_a
    h_ = tf.exp(th) * h_a
    # Get box corners
    ul_x = tf.maximum(0, tf.to_int32(tf.floor(-w_/2+x_)))
    ul_y = tf.maximum(0, tf.to_int32(tf.floor(-h_/2+y_)))
    br_x = tf.minimum(inp_res-1, tf.to_int32(tf.ceil(w_/2+x_)))
    br_y = tf.minimum(inp_res-1, tf.to_int32(tf.ceil(h_/2+y_)))

    return tf.to_int32(tf.stack([ul_x, ul_y, br_x, br_y], 1))

def iou(gt_box, pred_boxes):
    # (https://github.com/danfeiX/scene-graph-TF-release/)
    gt_box = gt_box.reshape((4))
    pred_boxes = pred_boxes.reshape((-1,4))

    # computer Intersection-over-Union between two sets of boxes
    ixmin = np.maximum(gt_box[0], pred_boxes[:,0])
    iymin = np.maximum(gt_box[1], pred_boxes[:,1])
    ixmax = np.minimum(gt_box[2], pred_boxes[:,2])
    iymax = np.minimum(gt_box[3], pred_boxes[:,3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) +
            (pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) *
            (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def tf_iou(gt_boxes, pred_boxes):
    # Modified IOU calculation using tensorflow ops

    # Reshape and tile boxes
    gt_boxes = tf.to_float(tf.reshape(gt_boxes,[-1,1,4]))
    pred_boxes = tf.to_float(tf.reshape(pred_boxes,[1,-1,4]))
    n_gt = tf.shape(gt_boxes)[0]
    n_pred = tf.shape(pred_boxes)[1]
    gt_boxes = tf.tile(gt_boxes, [1,n_pred,1])
    pred_boxes = tf.tile(pred_boxes, [n_gt,1,1])
    
    # Compute IOU between two sets of boxes
    ixmin = tf.maximum(gt_boxes[:,:,0], pred_boxes[:,:,0])
    iymin = tf.maximum(gt_boxes[:,:,1], pred_boxes[:,:,1])
    ixmax = tf.minimum(gt_boxes[:,:,2], pred_boxes[:,:,2])
    iymax = tf.minimum(gt_boxes[:,:,3], pred_boxes[:,:,3])

    iw = tf.maximum(ixmax - ixmin + 1, 0.)
    ih = tf.maximum(iymax - iymin + 1, 0.)
    inters = iw * ih

    # union
    uni = ((gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1.) * (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1.) +
            (pred_boxes[:,:,2] - pred_boxes[:,:,0] + 1.) *
            (pred_boxes[:,:,3] - pred_boxes[:,:,1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


# =============================================================================
# Postprocessing to evaluate scene graph predictions
# =============================================================================

def compare_classes(gt_classes, pred_classes):
    # Compare all pairs between gt_classes and pred_classes

    n = tf.shape(pred_classes)[1] if len(pred_classes.shape) == 2 else 1
    
    # Reshape and tile class predictions
    gt_classes = tf.reshape(gt_classes, [-1,1,n])
    pred_classes = tf.reshape(pred_classes, [1,-1,n])
    n_gt = tf.shape(gt_classes)[0]
    n_pred = tf.shape(pred_classes)[1]
    gt_classes = tf.tile(gt_classes, [1,n_pred,1])
    pred_classes = tf.tile(pred_classes, [n_gt,1,1])

    return tf.reduce_prod(tf.to_int32(tf.equal(gt_classes, pred_classes)), 2)

def _relation_recall(gt_triplets, pred_triplets,
                     gt_boxes, pred_boxes, iou_thresh, rel_filter=None):

    # compute the R@K metric for a set of predicted triplets

    # slightly modified from https://github.com/danfeiX/scene-graph-TF-release/
    # only change is the optional rel_filter

    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0

    for gt, gt_box in zip(gt_triplets, gt_boxes):
        if rel_filter is None or gt[1] == rel_filter:
            keep = np.zeros(pred_triplets.shape[0]).astype(bool)
            for i, pred in enumerate(pred_triplets):
                if gt[0] == pred[0] and gt[1] == pred[1] and gt[2] == pred[2]:
                    keep[i] = True
            if not np.any(keep):
                continue
            boxes = pred_boxes[keep,:]
            sub_iou = iou(gt_box[:4], boxes[:,:4])
            obj_iou = iou(gt_box[4:], boxes[:,4:])
            inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                                  np.where(obj_iou >= iou_thresh)[0])
            if inds.size > 0:
                num_correct_pred_gt += 1
        else:
            num_gt -= 1

    if num_gt > 0:
        return float(num_correct_pred_gt) / float(num_gt)
    else:
        return -1

def tf_relation_recall(gt_triplets, pred_triplets,
                       gt_boxes, pred_boxes, iou_thresh=0.5):
    iou_1 = tf.to_int32(tf_iou(gt_boxes[:,:4], pred_boxes[:,:4]) >= iou_thresh)
    iou_2 = tf.to_int32(tf_iou(gt_boxes[:,4:], pred_boxes[:,4:]) >= iou_thresh)
    classes = compare_classes(gt_triplets, pred_triplets)
    match = iou_1 * iou_2 * classes
    
    return tf.reduce_mean(tf.to_float(tf.reduce_max(match, 1))), match
