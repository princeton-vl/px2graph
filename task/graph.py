import tensorflow as tf
import numpy as np
import math
import scipy.misc

from px2graph.task.base import Task as ParentTask
from px2graph.util import img, calc
from px2graph.util.rcnn.generate_anchors import generate_anchors

class Task(ParentTask):
    """ Pixels to graph task functions.

    This class defines functions to generate training samples, process network
    output, calculate the loss, and evaluate prediction accuracy.

    There is support for three training and evaluation settings:
        PR: Predicate classification (object bounding boxes and categories provided)
        CL: Object + predicate classification (object bounding boxes provided)
        SG: Full scene graph generation (nothing but the image provided at test time)

    """

    proc_arg_dtype = [tf.float32, tf.float32, tf.int32, tf.int32, tf.int32,
                      tf.float32, tf.int32, tf.int32, tf.int32]
    data_arg_dtype = proc_arg_dtype
    np_dtype = [d.as_numpy_dtype for d in proc_arg_dtype]
    num_inputs = 3
    idx_ref = {'inp_img':0, 'inp_hm':1, 'kp_ref':2, 'num_kp_ref':3, 'node_classes':4,
               'node_bboxes':5, 'rel_ref':6, 'obj_ref':7, 'sample_idx':8}

    anchors = generate_anchors(base_size=16, scales=2**np.arange(0,5)) - 7.5
    w_a_ref = (anchors[:,2] - anchors[:,0]).astype(np.float32)
    h_a_ref = (anchors[:,3] - anchors[:,1]).astype(np.float32)

    def __init__(self, opt, ds):
        super(Task, self).__init__(opt, ds)

        # Update options with info for defining model
        opt.num_hm_channels = 2
        opt.num_cats = ds.num_cats
        opt.num_rels = ds.num_rels
        opt.max_nodes = max(ds.max_objs, ds.max_total_rels)
        opt.bbox_channels = 4 * len(self.anchors)

        if opt.sg_task == 'PR': opt.det_inputs = 2*opt.num_cats
        else: opt.det_inputs = 2*len(self.anchors)

        self.data_shape = [
            [opt.input_res, opt.input_res, 3],
            [opt.output_res, opt.output_res, opt.det_inputs],
            [2, opt.max_nodes, 2],
            [4],
            [opt.max_nodes, opt.obj_slots + opt.rel_slots],
            [opt.max_nodes, opt.obj_slots, 5],
            [opt.max_nodes, opt.rel_slots, 2],
            [opt.max_nodes, 7],
            [1]
        ]

    def load_sample_data(self, idx):
        """ Generate a training sample.

        Args:
            idx: Array of two integers, first value is the sample index, second
                value is a training flag (1 for training, 0 for validation). If
                the training flag is set, then data augmentation will be applied
                to the sample.

        Returns:
            sample: A list containing input and label information as follows:
                0 : inp_img (in_res x in_res x 3)
                        Fixed-size input image
                1 : inp_hm (out_res x out_res x ?)
                        Ground truth or RPN bounding boxes
                2 : kp_ref (2 x max_nodes x 2)
                        Keypoint locations indicating center of each node and
                        relationship, used to extract features from correct
                        location when applying loss.
                3 : num_kp_ref (2)
                        Indicates how many ground truth objects and relationships
                        are annotated in a given sample.
                4 : node_classes (max_nodes x obj_slots + rel_slots)
                        Object and relationship class assignment
                5 : node_bboxes (max_nodes x obj_slots x 5)
                        Bounding bbox info: anchor box choice, tx, ty, tw, th
                6 : rel_ref (max_nodes x rel_slots x 2)
                        Indices of source and destination node for each
                        relationship
                7 : obj_ref: (max_nodes x 7)
                        Node information reference includes kp_idx, slot_idx,
                        class, box coords
                8 : sample_idx: (1)
        """

        ### ====================================================================
        ### Initial set up
        ### ====================================================================

        idx, train_flag = idx
        ds, opt = self.ds, self.opt

        # Initialize empty tensors
        sample = [np.zeros(d, self.np_dtype[i]) for i,d in enumerate(self.data_shape)]
        inp_hm, kp_ref, num_kp_ref, node_classes, node_bboxes, rel_ref, obj_ref = sample[1:-1]
        sample[-1][0] = idx

        # Load input image
        inp_img = ds.load_image(idx)
        ht, wd = inp_img.shape[:2]
        center, scale = [wd/2, ht/2], max(ht, wd) / 200
        do_flip = False

        # Data augmentation
        if train_flag:
            t_ = 40 * scale
            center[0] += np.random.randint(-t_, t_+1)
            center[1] += np.random.randint(-t_, t_+1)
            scale *= 2**(np.random.rand()*1.5 - 1)
            do_flip = np.random.rand() > .5
            if do_flip: inp_img = np.flip(inp_img, 1)

        sample[0] = img.crop(inp_img, center, scale, [opt.input_res]*2)
        inp_res = [opt.input_res] * 2
        out_res = [opt.output_res] * 2

        # Load sample info
        tmp_info = ds.preload_sample_info(idx)
        obj_classes = ds.get_obj_classes(idx, tmp_info)
        obj_bboxes = ds.get_bboxes(idx, tmp_info)
        obj_rels = ds.get_rels(idx, tmp_info)

        # Various things to track as we set up the sample
        obj_kp_count, obj_slot_ref = 0, {}
        rel_kp_count, rel_slot_ref = 0, {}
        tmp_idx_ref = [] # Track which objects are not cropped out

        # All indices (used to efficiently sample negative pixels)
        all_idxs = np.unravel_index(np.arange(opt.output_res**2), out_res)
        all_idxs = np.array(all_idxs).T
        free_obj_kps = np.ones(opt.output_res**2).astype(bool)
        free_rel_kps = np.ones(opt.output_res**2).astype(bool)

        # Fill reference data with -1
        for s_arr in sample[3:-1]: s_arr.fill(-1)

        ### ====================================================================
        ### Objects
        ### ====================================================================

        if obj_bboxes is not None:
            if do_flip:
                obj_bboxes = calc.flip_boxes(obj_bboxes, wd)
                # Account for any relationship labels that need to flip
                for i in range(len(obj_rels)):
                    for j in range(len(obj_rels[i])):
                        obj_rels[i][j][1] = ds.rel_flip_ref[obj_rels[i][j][1]]

            n_objs = len(obj_classes)

            for i in range(n_objs):
                # Transform bounding box coordinates
                ul = img.transform(obj_bboxes[i][0], center, scale, inp_res)
                br = img.transform(obj_bboxes[i][1], center, scale, inp_res)

                # Check that at least part of box falls inbounds
                if calc.box_inbounds([ul,br], inp_res):
                    # Adjust box in case a corner is out of bounds
                    ul = np.maximum(np.zeros(2), ul).astype(int)
                    br = np.minimum(np.array(inp_res)-1, br).astype(int)
                    pt = (ul + br) / 2

                    # Transform from input to output resolution
                    out_in_ratio = opt.output_res / opt.input_res
                    pt_out = (pt * out_in_ratio).astype(int)
                    pt_idx = np.ravel_multi_index(pt_out, out_res)

                    # Calculate anchor box assignment plus adjustments
                    adj_box = np.array([ul - pt, br - pt])
                    anchor_idx = calc.iou(adj_box, self.anchors).argmax()
                    anch_box = self.anchors[anchor_idx]

                    x_a, y_a = (pt_out + .5) / out_in_ratio
                    w_a, h_a = anch_box[2]-anch_box[0], anch_box[3]-anch_box[1]
                    tx, ty = (pt[0] - x_a) / w_a, (pt[1] - y_a) / h_a
                    tw = np.log((br[0] - ul[0] + 1) / w_a)
                    th = np.log((br[1] - ul[1] + 1) / h_a)

                    if free_obj_kps[pt_idx]:
                        # First object to appear at this location
                        free_obj_kps[pt_idx] = False
                        obj_slot_ref[pt_idx] = [0, obj_kp_count]
                        kp_ref[0, obj_kp_count] = pt_out
                        obj_kp_count += 1

                    if obj_slot_ref[pt_idx][0] < opt.obj_slots:
                        slot_idx, kp_idx = obj_slot_ref[pt_idx]

                        # Prepare input heatmap
                        if opt.sg_task != 'SG':
                            inp_idx = 2 * anchor_idx
                            if opt.sg_task == 'PR': inp_idx = 2 * obj_classes[i]

                            # Input heatmap has two channels to represent a box,
                            # to indicate the center of box and the full mask
                            inp_hm[pt_out[1], pt_out[0], inp_idx] = 1

                            ul_ = (ul * out_in_ratio).astype(int)
                            br_ = (br * out_in_ratio).astype(int)
                            inp_hm[ul_[1]:br_[1], ul_[0]:br_[0], inp_idx+1] = 1

                        # Update output reference
                        node_classes[kp_idx, slot_idx] = obj_classes[i]
                        node_bboxes[kp_idx, slot_idx] = anchor_idx, tx, ty, tw, th

                        # Update object reference
                        tmp_idx_ref += [i]
                        obj_ref[len(tmp_idx_ref) - 1] = kp_idx, slot_idx, obj_classes[i], \
                                                        ul[0], ul[1], br[0], br[1]
                        obj_slot_ref[pt_idx][0] += 1

        if opt.sg_task == 'SG' and opt.use_rpn:
            # Set up RPN input if we're using box proposals
            max_rpn_boxes = 100
            rpn_boxes, rpn_scores = ds.get_rpn_proposals(idx)

            # Perform NMS on proposals
            nms_dets = calc.py_cpu_nms(np.concatenate([rpn_boxes,rpn_scores],1), .4)
            rpn_boxes = rpn_boxes[nms_dets][:max_rpn_boxes]
            rpn_scores = rpn_scores[nms_dets][:max_rpn_boxes]
            num_rpn_boxes = rpn_boxes.shape[0]

            rpn_boxes = rpn_boxes.reshape((-1,2,2))
            if do_flip: rpn_boxes = calc.flip_boxes(rpn_boxes, wd)

            # Loop through proposals, setting up input heatmap
            for i in range(num_rpn_boxes):
                # Transform coordinates
                ul = img.transform(rpn_boxes[i][0], center, scale, inp_res)
                br = img.transform(rpn_boxes[i][1], center, scale, inp_res)

                # Check that at least part of box falls inbounds
                if calc.box_inbounds([ul,br], inp_res):
                    # Adjust corners
                    ul = np.maximum(np.zeros(2), ul).astype(int)
                    br = np.minimum(np.array(inp_res)-1, br).astype(int)
                    pt = (ul + br) / 2

                    # Transform from input to output resolution
                    out_in_ratio = opt.output_res / opt.input_res
                    pt_out = (pt * out_in_ratio).astype(int)

                    # Calculate anchor box assignment
                    adj_box = np.array([ul - pt, br - pt])
                    anchor_idx = calc.iou(adj_box, self.anchors).argmax()

                    # Draw on input heatmap
                    inp_idx = 2 * anchor_idx
                    inp_hm[pt_out[1], pt_out[0], 1]
                    ul_ = (ul * out_in_ratio).astype(int)
                    br_ = (br * out_in_ratio).astype(int)
                    inp_hm[ul_[1]:br_[1], ul_[0]:br_[0], inp_idx+1] = 1

        # Get negative indices
        tmp_neg_idxs = np.arange(opt.output_res**2)[free_obj_kps]
        np.random.shuffle(tmp_neg_idxs)
        tmp_neg_idxs = all_idxs[tmp_neg_idxs[:opt.max_nodes-obj_kp_count]]
        kp_ref[0, obj_kp_count:] = tmp_neg_idxs

        ### ====================================================================
        ### Relationships
        ### ====================================================================

        rel_count = 0
        if obj_bboxes is not None:
            for subj_idx in range(len(tmp_idx_ref)):
                for rel in obj_rels[tmp_idx_ref[subj_idx]]:
                    # rel is a tuple: (object index, relationship index)
                    if rel[0] in tmp_idx_ref:
                        obj_idx = tmp_idx_ref.index(rel[0])

                        # Get midpoint of both objects
                        pt1 = kp_ref[0, obj_ref[subj_idx][0]]
                        pt2 = kp_ref[0, obj_ref[obj_idx][0]]
                        pt = ((pt1+pt2)/2).astype(int)
                        pt_idx = np.ravel_multi_index(pt, out_res)

                        if free_rel_kps[pt_idx]:
                            # First relationship to appear at this location
                            free_rel_kps[pt_idx] = False
                            rel_slot_ref[pt_idx] = [0, rel_kp_count]
                            kp_ref[1, rel_kp_count] = pt
                            rel_kp_count += 1

                        if rel_slot_ref[pt_idx][0] < opt.rel_slots:
                            # If we still have slots left
                            slot_idx, kp_idx = rel_slot_ref[pt_idx]

                            node_classes[kp_idx, opt.obj_slots+slot_idx] = rel[1]
                            rel_ref[kp_idx, slot_idx] = [subj_idx, obj_idx]

                            rel_slot_ref[pt_idx][0] += 1
                            rel_count += 1

        # Get negative indices
        tmp_neg_idxs = np.arange(opt.output_res**2)[free_rel_kps]
        np.random.shuffle(tmp_neg_idxs)
        tmp_neg_idxs = all_idxs[tmp_neg_idxs[:opt.max_nodes-rel_kp_count]]
        kp_ref[1, rel_kp_count:] = tmp_neg_idxs

        num_kp_ref[:] = [obj_kp_count, rel_kp_count, len(tmp_idx_ref), rel_count]

        return sample

    def preprocess(self, data, train_flag):
        # Augment color/brightness/etc
        #augment_img = tf.clip_by_value(tf.image.random_saturation(
        #    tf.image.random_contrast(
        #    tf.image.random_brightness(
        #    tf.image.random_hue(data[0], .2), .3), .5, 1.5), .5, 1.5), 0, 1)

        return data

    def loss(self, net, label, sample_idx):
        
        self.loss_ref = [
            'obj_hm',
            'obj_class',
            'obj_bbox_anch',
            'obj_bbox_reg',
            'rel_hm',
            'rel_class',
            'tag_pull',
            'tag_push'
        ]

        batchsize = int(net[0].shape[0])
        num_anchors = len(self.anchors)
        opt = self.opt

        # Take first half of network output which is collected from ground truth
        # keypoint locations so that we can apply the loss correctly
        tmp_net = [(net[i][:,:opt.max_nodes] if i not in [0,12] else net[i])
                   for i in range(len(net))]
        tmp_net[12] = tmp_net[12][:,:,:opt.max_nodes]

        # Unpack network output
        hm_acts = tmp_net[0]
        obj_acts, obj_scores, obj_classes, node_tags, bbox_anch, bbox_reg = tmp_net[1:7]
        rel_acts, rel_scores, rel_classes, sbj_tags, obj_tags, kp_ref = tmp_net[7:]
        # Unpack ground truth labels
        num_kp_ref, classes_gt, bbox_gt, rels_gt, obj_ref = label

        # Apply loss to each sample independently
        all_losses, all_matched = [], []
        for i in range(batchsize):
            # Get reference information
            num_objs = num_kp_ref[i][0]
            num_rels = num_kp_ref[i][1]

            def calc_obj_loss():

                # -------------------------------------------------------------
                # Prepare data for calculating the loss
                # -------------------------------------------------------------

                split_ref = [1, opt.num_cats, opt.tag_dim, opt.bbox_channels, opt.bbox_channels]

                def obj_kp_matchup(kp_idx):
                    # Get predictions and ground truth comparison
                    tmp_preds = [tf.nn.softmax(obj_classes[i,kp_idx]),
                                 tf.nn.softmax(bbox_anch[i,kp_idx,:,:num_anchors])]
                    tmp_preds = tf.concat(tmp_preds, 1)

                    tmp_gt = [tf.one_hot(classes_gt[i,kp_idx,:opt.obj_slots], opt.num_cats),
                              tf.one_hot(tf.to_int32(bbox_gt[i,kp_idx,:,0]), num_anchors)]
                    tmp_gt = tf.concat(tmp_gt, 1)

                    # Shuffle to best matchup
                    obj_info = tf.concat([v[i,kp_idx] for v in tmp_net[2:7]], 1)
                    return calc.match_and_shuffle(tmp_preds, tmp_gt, obj_info, split_ref)

                matched_objs = tf.map_fn(obj_kp_matchup, tf.range(num_objs),
                                         [tf.float32]*len(split_ref))

                # Pad and flatten predictions
                matched_objs = [tf.pad(tmp_val, [[0,opt.max_nodes-num_objs],[0,0],[0,0]])
                                for tmp_val in matched_objs]
                matched_objs = [tf.reshape(tmp_val, [opt.max_nodes*opt.obj_slots,-1])
                                for tmp_val in matched_objs]
                obj_scores_, obj_classes_, tags_, bbox_anch_, bbox_reg_ = matched_objs

                # Flatten ground truth
                obj_classes_gt = tf.reshape(classes_gt[i,:,:opt.obj_slots], [-1])
                obj_bbox_gt = tf.reshape(bbox_gt[i], [opt.max_nodes*opt.obj_slots, -1])
                obj_scores_gt = tf.to_float(tf.greater(obj_classes_gt, -1))

                # Filter out non-objects
                filter_idxs = tf.squeeze(tf.where(obj_bbox_gt[:,0] > -1), 1)
                tmp_gt_bbox = tf.gather(obj_bbox_gt[:], filter_idxs)
                tmp_pred_bbox = tf.gather(bbox_reg_, filter_idxs)
                tmp_num_objs = tf.shape(tmp_gt_bbox)[0]

                # Rearrange bbox regressed values to (num_objs x num_anchors x 4)
                tmp_pred_bbox = tf.reshape(tmp_pred_bbox, [tmp_num_objs, num_anchors, 4])
                # Gather values corresponding to correct anchor box
                anchor_idxs = tf.stack([tf.range(tmp_num_objs), tf.to_int32(tmp_gt_bbox[:,0])],1)
                tmp_pred_bbox = tf.gather_nd(tmp_pred_bbox, anchor_idxs)

                # -------------------------------------------------------------
                # Calculate each loss term
                # -------------------------------------------------------------

                # Heatmap activations and per-slot score loss
                hm_act_loss = calc.activation_loss(obj_acts[i], num_objs)
                score_loss = calc.bce_loss(tf.squeeze(obj_scores_[:opt.obj_slots*num_objs],1),
                                           obj_scores_gt[:opt.obj_slots*num_objs])
                # Bounding box losses
                bbox_anch_loss = calc.classify_loss(bbox_anch_[:,:num_anchors],
                                                    tf.to_int32(obj_bbox_gt[:,0]))
                bbox_reg_loss = calc.huber_loss(tmp_pred_bbox,tmp_gt_bbox[:,1:])
                # Object class loss
                class_loss = calc.classify_loss(obj_classes_, obj_classes_gt)

                obj_loss = [
                    hm_act_loss + score_loss,
                    class_loss,
                    bbox_anch_loss,
                    bbox_reg_loss,
                ]
                
                return obj_loss, matched_objs

            def calc_rel_loss():

                # -------------------------------------------------------------
                # Prepare data for calculating the loss
                # -------------------------------------------------------------

                obj_scores_, obj_classes_, tags_, bbox_anch_, bbox_reg_ = matched_o
                split_ref = [1, opt.num_rels, opt.tag_dim, opt.tag_dim]

                def get_tags(kp_idx, ref_idx):
                    # Collect tags from object keypoint locations
                    obj_idxs = rels_gt[i,kp_idx,:,ref_idx]
                    tmp_idxs = tf.maximum(obj_idxs, tf.zeros(tf.shape(obj_idxs), tf.int32))
                    tmp_info = tf.gather(obj_ref[i], tmp_idxs)
                    kp_idxs, slot_idxs = tmp_info[:,0], tmp_info[:,1]
                    tmp_idxs = kp_idxs*opt.obj_slots + slot_idxs
                    tmp_tags = tf.gather(tags_, tmp_idxs)
                    tmp_tags = tf.where(obj_idxs > -1, tmp_tags, tf.zeros(tf.shape(tmp_tags)))
                    return tmp_tags

                def rel_kp_matchup(kp_idx):
                    # Get predictions and ground truth comparison
                    tmp_preds = tf.concat([tf.nn.softmax(rel_classes[i,kp_idx]),
                                           sbj_tags[i,kp_idx], obj_tags[i,kp_idx]], 1)

                    tmp_gt = [tf.one_hot(classes_gt[i,kp_idx, opt.obj_slots:], opt.num_rels),
                              get_tags(kp_idx, 0), get_tags(kp_idx, 1)]
                    tmp_gt = tf.concat(tmp_gt, 1)

                    # Shuffle to best matchup
                    rel_info = tf.concat([v[i,kp_idx] for v in tmp_net[8:12]], 1)
                    return calc.match_and_shuffle(tmp_preds, tmp_gt, rel_info, split_ref)

                matched_rels = tf.map_fn(rel_kp_matchup, tf.range(num_rels),
                                         [tf.float32]*len(split_ref))

                # Pad and flatten predictions
                matched_rels = [tf.pad(tmp_val, [[0,opt.max_nodes-num_rels],[0,0],[0,0]])
                                for tmp_val in matched_rels]
                matched_rels = [tf.reshape(tmp_val, [opt.max_nodes*opt.rel_slots,-1])
                                for tmp_val in matched_rels]
                rel_scores_, rel_classes_, sbj_tags_, obj_tags_ = matched_rels

                # Flatten ground truth
                rel_classes_gt = tf.reshape(classes_gt[i,:,opt.obj_slots:], [-1])
                rel_scores_gt = tf.to_float(tf.greater(rel_classes_gt, -1))

                # Prepare inputs for associative embedding loss
                tmp_obj_ref = tf.size(tf.where(obj_ref[i,:,0] > -1))
                tmp_obj_ref = obj_ref[i, :tmp_obj_ref]
                tag_ref = tf.gather(tags_, tmp_obj_ref[:,0]*opt.obj_slots + tmp_obj_ref[:,1])

                tag_idxs = [tf.reshape(rels_gt[i,:,:,0],[opt.max_nodes*opt.rel_slots,1]),
                            tf.reshape(rels_gt[i,:,:,1],[opt.max_nodes*opt.rel_slots,1])]
                tag_idxs = tf.concat(tag_idxs, 0)
                tag_preds = tf.concat([sbj_tags_, obj_tags_], 0)

                filter_idxs = tf.squeeze(tf.where(tf.squeeze(tag_idxs,1) > -1), 1)
                tag_idxs = tf.gather(tag_idxs, filter_idxs)
                tag_preds = tf.gather(tag_preds, filter_idxs)

                # -------------------------------------------------------------
                # Calculate each loss term
                # -------------------------------------------------------------

                # Heatmap activations and per-slot score loss
                hm_act_loss = calc.activation_loss(rel_acts[i], num_rels)
                score_loss = calc.bce_loss(tf.squeeze(rel_scores_[:opt.obj_slots*num_objs],1),
                                           rel_scores_gt[:opt.obj_slots*num_objs])
                # Relationship class loss
                class_loss = calc.classify_loss(rel_classes_, rel_classes_gt)
                # Associative embedding loss
                tag_loss = calc.push_pull_loss(tag_idxs, tag_preds, tag_ref,
                                               tag_margin=opt.tag_margin)

                rel_loss = [
                    hm_act_loss + score_loss,
                    class_loss,
                    *tag_loss
                ]
                
                return rel_loss, matched_rels

            def dummy_vals(x):
                tmp_loss = [tf.zeros([])] * 4
                num_slots = opt.rel_slots if x else opt.obj_slots
                tmp_vals = tmp_net[8:12] if x else tmp_net[2:7]
                tmp_vals = [tf.reshape(v[i], [opt.max_nodes*num_slots,-1]) for v in tmp_vals]

                return tmp_loss, tmp_vals

            o_loss, matched_o = tf.cond(num_objs > 0, calc_obj_loss, lambda: dummy_vals(0))
            r_loss, matched_r = tf.cond(num_rels > 0, calc_rel_loss, lambda: dummy_vals(1))

            all_losses += [o_loss + r_loss]
            all_matched += [matched_o + matched_r]

        loss = []
        for i in range(len(all_losses[0])):
            tmp_loss = []
            for j in range(batchsize):
                tmp_loss += [all_losses[j][i]]
            loss += [tf.reduce_mean(tf.stack(tmp_loss,0))]

        self.__aux_data = []
        for i in range(len(all_matched[0])):
            tmp_match = []
            for j in range(batchsize):
                tmp_match += [all_matched[j][i]]
            self.__aux_data += [tf.stack(tmp_match,0)]

        return loss

    def postprocess(self, net, label, sample_idx):

        batchsize = int(net[0].shape[0])
        num_anchors = len(self.anchors)
        opt = self.opt

        if opt.sg_task == 'SG':
            # For SG, take last half of network output which is collected from
            # predicted keypoint locations (no ground truth information is used)
            tmp_net = [(net[i][:,opt.max_nodes:] if i not in [0,12] else net[i])
                       for i in range(len(net))]
            tmp_net[12] = tmp_net[12][:,:,opt.max_nodes:]
        else:
            # For PR and CL tasks use ground truth object locations
            tmp_net = [(net[i][:,:opt.max_nodes] if i not in [0,12] else net[i])
                       for i in range(len(net))]
            tmp_net[12] = tmp_net[12][:,:,:opt.max_nodes]

        for i in [1,7]:
            # Tile heatmap activations across slots
            tmp_net[i] = tf.tile(tf.expand_dims(tf.expand_dims(tmp_net[i], 2), 3),
                                 [1, 1, opt.obj_slots if i == 1 else opt.rel_slots, 1])
        for i in range(1,12):
            # Flatten predictions across slots
            last_dim = int(tmp_net[i].shape[-1])
            tmp_net[i] = tf.reshape(tmp_net[i], [batchsize, -1, last_dim])

            # Apply sigmoid to all logits
            if i in [1,2,7,8]: tmp_net[i] = tf.nn.sigmoid(tmp_net[i])

        # Unpack network output
        hm_acts = tmp_net[0]
        obj_acts, obj_scores, obj_classes, node_tags, bbox_anch, bbox_reg = tmp_net[1:7]
        obj_scores_, obj_classes_, tags_, bbox_anch_, bbox_reg_ = self.__aux_data[:5]
        rel_acts, rel_scores, rel_classes, sbj_tags, obj_tags, kp_ref = tmp_net[7:]
        # Unpack ground truth labels
        num_kp_ref, classes_gt, bbox_gt, rels_gt, obj_ref = label

        all_objs, all_rels = [], []
        pred_obj_count, pred_rel_count = [], []
        n_found_objs, n_found_rels, obj_count, rel_count = 0, 0, 0, 0

        for i in range(batchsize):
            # Ground truth objects
            n_gt_objs = tf.size(tf.where(obj_ref[i,:,0] > -1))
            obj_cat_gt = obj_ref[i,:n_gt_objs,2]
            obj_bbox_gt = obj_ref[i,:n_gt_objs,3:]

            # Raw gt values matched up when applying loss
            obj_cat_gt_ = tf.reshape(classes_gt[i,:,:opt.obj_slots], [-1])
            obj_bbox_gt_ = tf.reshape(bbox_gt[i], [opt.max_nodes*opt.obj_slots, -1])
            filter_idxs = tf.squeeze(tf.where(obj_bbox_gt_[:,0] > -1), 1)
            obj_cat_gt_ = tf.gather(obj_cat_gt_[:], filter_idxs)
            obj_bbox_gt_ = tf.gather(obj_bbox_gt_[:], filter_idxs)

            # Ground truth relationships
            rel_idxs_gt = tf.where(classes_gt[i,:,opt.obj_slots:] > -1)
            n_gt_rels = tf.shape(rel_idxs_gt)[0]
            rel_class_gt = tf.gather_nd(classes_gt[i,:,opt.obj_slots:], rel_idxs_gt)
            rel_ref_gt = tf.gather_nd(rels_gt[i], rel_idxs_gt)

            def get_top_detections(hm_acts, scores, attributes, keypoints, thr, n_slots):
                """ Collect top detections and their attributes

                Gather top object and relationship detections as defined by
                their heatmap activation and slot score.
                """

                t0,t1,t2 = thr # Define cutoff threshold for detections
                scores = tf.reshape((hm_acts - (t1-t0)) * (scores - (t2-t0)), [-1])
                scores, sorted_idxs = calc.top_k_threshold(scores, opt.max_nodes, t0)
                n_dets = tf.size(sorted_idxs)

                # Gather attributes from top detections
                vals = [tf.gather(v, sorted_idxs) for v in attributes]
                kp_idxs = tf.to_int32(sorted_idxs // n_slots)
                kps = tf.gather(keypoints, kp_idxs)

                return scores, vals, kps, n_dets

            # -------------------------------------------------------------
            # Object predictions
            # -------------------------------------------------------------

            if opt.sg_task in ['SG', 'CL']:
                n_ = n_gt_objs if opt.sg_task == 'CL' else opt.max_nodes
                n_ *= opt.obj_slots

                obj_vals = [v[i] for v in tmp_net[3:7]]
                tmp_obj_scores, obj_vals, tmp_kps, n_objs = get_top_detections( \
                    obj_acts[i,:n_], obj_scores[i,:n_], obj_vals, kp_ref[i,0], opt.obj_thr, opt.obj_slots)
                tmp_class, tmp_tag, tmp_anch, tmp_box_reg = obj_vals

            elif opt.sg_task == 'PR':
                # Use ground truth objects
                tmp_obj_scores = tf.nn.sigmoid(tf.squeeze(tf.gather(obj_scores_[i], filter_idxs),1))
                tmp_tag = tf.gather(tags_[i], filter_idxs)

                tmp_class = tf.one_hot(obj_cat_gt_, 150)
                tmp_anch = tf.one_hot(tf.to_int32(obj_bbox_gt_[:,0]), len(self.anchors))
                tmp_box_reg = tf.tile(obj_bbox_gt_[:,1:], [1,len(self.anchors)])

                kp_idxs = tf.to_int32(filter_idxs // opt.obj_slots)
                tmp_kps = tf.gather(kp_ref[i,0], kp_idxs)
                n_objs = tf.size(filter_idxs)

            k = 5

            def postprocess_objs():
                # Predicted classes and bounding boxes
                obj_cat_scores, obj_cat_topk = tf.nn.top_k(tmp_class, k=k)
                obj_bbox_pred = calc.tf_decode_boxes(self.anchors, tmp_kps,
                                                     tmp_anch, tmp_box_reg,
                                                     opt.input_res, opt.output_res)

                # Concatenate all object information
                to_concat = [tf.expand_dims(tmp_obj_scores, 1),
                             obj_cat_topk, obj_cat_scores, obj_bbox_pred, tmp_tag]
                objs = tf.concat([tf.to_float(v) for v in to_concat], 1)
                n_objs = tf.shape(objs)[0]

                # Do NMS on bounding boxes (per class)
                if opt.obj_box_nms > 0:
                    tmp_cats, tmp_idxs = tf.unique(obj_cat_topk[:,0])
                    n_cats = tf.shape(tmp_cats)[0]

                    def nms_single_class(ref_idx):
                        w_class = tf.squeeze(tf.where(tf.equal(tmp_idxs, ref_idx)),1)
                        bb_pred = tf.gather(tf.to_float(obj_bbox_pred), w_class)
                        o_score = tf.gather(tmp_obj_scores, w_class)
                        n_obj = tf.shape(bb_pred)[0]
                        nms_idxs = tf.image.non_max_suppression(bb_pred, o_score,
                                                                n_obj, iou_threshold=opt.obj_box_nms)
                        return tf.gather(w_class, nms_idxs)

                    # Collect nms idxs
                    ref_idxs = tf.range(n_cats)
                    cat_fn = lambda a, x: tf.concat([a, nms_single_class(x)], 0)
                    nms_box_idxs = tf.cond(n_cats > 0,
                                           lambda: tf.foldl(cat_fn, ref_idxs[1:], 
                                                            initializer=nms_single_class(ref_idxs[0])),
                                           lambda: tf.to_int64(ref_idxs))
                    n_objs = tf.shape(nms_box_idxs)[0]
                    nms_box_idxs,_ = tf.nn.top_k(-nms_box_idxs, k=n_objs)
                    nms_box_idxs = -nms_box_idxs
                    objs = tf.gather(objs, nms_box_idxs)

                n_objs = tf.minimum(opt.max_nodes, n_objs)
                objs = objs[:n_objs]
                obj_cat_pred = tf.to_int32(objs[:,1])
                obj_bbox_pred = objs[:,(2*k+1):(2*k+5)]

                # Calculate IOUs between predicted and ground truth boxes
                bbox_ious = calc.tf_iou(obj_bbox_gt, obj_bbox_pred)
                class_matchup = calc.compare_classes(obj_cat_gt, obj_cat_pred)
                matched_objs = tf.to_int32(tf.greater_equal(bbox_ious, .5)) * class_matchup
                gt_objs_found = tf.reduce_max(matched_objs, 1)

                # Objects that match up to a ground truth annotation (true positives)
                pred_obj_TP = tf.reduce_max(matched_objs, 0)
                objs = tf.concat([objs, tf.expand_dims(tf.to_float(pred_obj_TP),1)], 1)

                # Collect all ground truth objects that were missed
                missed_gt_idxs = tf.squeeze(tf.where(tf.equal(gt_objs_found, 0)),1)
                n_missed = tf.shape(missed_gt_idxs)[0]

                missed_obj_cats = tf.gather(obj_cat_gt, missed_gt_idxs)
                missed_obj_bbox = tf.gather(obj_bbox_gt, missed_gt_idxs)
                to_concat = [tf.zeros([n_missed,1]), tf.expand_dims(missed_obj_cats,1),
                             tf.zeros([n_missed,2*k-1]), missed_obj_bbox,
                             tf.zeros([n_missed,opt.tag_dim]), 2*tf.ones([n_missed,1])]

                missed_objs = tf.concat([tf.to_float(v) for v in to_concat], 1)
                objs = tf.concat([objs, missed_objs], 0)

                # Get indices to map gt obj idxs to the proper locations in 'objs'
                tmp_idxs = tf.range(tf.shape(missed_gt_idxs)[0]) + n_objs
                missed_idx_ref = tf.to_int64(tf.scatter_nd(tf.expand_dims(missed_gt_idxs,1),
                                                           tf.expand_dims(tmp_idxs,1),
                                                           [tf.to_int64(n_gt_objs),1]))

                pred_idx_ref = tf.to_int32(tf.where(gt_objs_found > 0,
                                                    tf.argmax(matched_objs, 1),
                                                    tf.squeeze(missed_idx_ref, 1)))

                return objs, n_objs, gt_objs_found, pred_idx_ref

            def dummy_obj_return():
                to_concat = [tf.zeros([n_gt_objs,1]), tf.expand_dims(obj_cat_gt,1),
                             tf.zeros([n_gt_objs,2*k-1]), obj_bbox_gt,
                             tf.zeros([n_gt_objs,opt.tag_dim]), 2*tf.ones([n_gt_objs,1])]
                objs = tf.concat([tf.to_float(v) for v in to_concat], 1)
                gt_objs_found = tf.zeros([n_gt_objs], tf.int32)
                pred_idx_ref = tf.range(n_gt_objs, dtype=tf.int32)

                return objs, tf.zeros([], tf.int32), gt_objs_found, pred_idx_ref

            objs, n_objs, gt_objs_found, pred_idx_ref = tf.cond(n_objs > 0,
                                                                postprocess_objs,
                                                                dummy_obj_return)
            obj_cat_pred = tf.to_int32(objs[:n_objs, 1])
            obj_bbox_pred = objs[:n_objs, (2*k+1):(2*k+5)]
            obj_tag_ref = objs[:n_objs, -opt.tag_dim-1:-1]

            # -------------------------------------------------------------
            # Relationship predictions
            # -------------------------------------------------------------

            rel_vals = [v[i] for v in tmp_net[9:12]]
            tmp_scores_aux, rel_vals, _, n_rels = get_top_detections( \
                    rel_acts[i], rel_scores[i], rel_vals, kp_ref[i,0], opt.rel_thr, opt.rel_slots)
            k = opt.rel_top_k
            
            def postprocess_rels():
                tmp_rel_class, sbj_tag, obj_tag = rel_vals
                rel_cat_scores, rel_cat_topk = tf.nn.top_k(tmp_rel_class, k=k)
                rel_cat_scores = tf.nn.sigmoid(rel_cat_scores)

                def matchup(tag_ref, tag_pred):
                    # n_rels x n_objs x tag_dim
                    tag_pred = tf.tile(tf.expand_dims(tag_pred,1),  [1,n_objs,1])
                    tag_dist = tf.norm(tag_ref - tag_pred, axis=2)
                    return tf.to_int32(tf.argmin(tag_dist, 1))

                tag_ref = tf.tile(tf.expand_dims(obj_tag_ref,0), [n_rels,1,1])
                sbj_idx = matchup(tag_ref, sbj_tag)
                obj_idx = matchup(tag_ref, obj_tag)

                # Use top-k class predictions
                tile_ = lambda x: tf.tile(tf.expand_dims(x, 1), [1,k])
                tmp_scores = tile_(tmp_scores_aux)
                score_offset = tf.to_float(tf.meshgrid(tf.range(k), tf.range(n_rels))[0])
                t_ = opt.class_thr
                tmp_scores = tmp_scores * t_[0] + \
                             rel_cat_scores * t_[1] + \
                             t_[2] * (t_[3] - score_offset)
                tmp_scores = tf.reshape(tmp_scores, [n_rels*k])

                # Find unique tuples
                M, K = int(1e6), int(1e3)
                t = tf.stack([tile_(sbj_idx), rel_cat_topk, tile_(obj_idx)], 2)
                t = tf.reshape(t, [n_rels*k, 3])
                t, idx_ref = tf.unique(t[:,0]*M + t[:,1]*K + t[:,2])
                tmp_tuples = tf.stack([t // M, (t // K ) % K, t % K], 1)

                # Get corresponding scores, sort to get top 100
                idx_ref, shuffle_ref = tf.nn.top_k(-idx_ref, k=tf.size(idx_ref))
                tmp_scores = tf.segment_max(tf.gather(tmp_scores, shuffle_ref), -idx_ref)
                tmp_scores, sorted_idxs = tf.nn.top_k(tmp_scores, k=tf.size(tmp_scores))
                tmp_scores = tmp_scores[:100]
                tmp_tuples = tf.gather(tmp_tuples, sorted_idxs)[:100]

                # Set up triplets and boxes for calculating recall @ 100
                def get_triplets(class_ref, bbox_ref, tuples):
                    sbj_class = tf.gather(class_ref, tuples[:,0])
                    obj_class = tf.gather(class_ref, tuples[:,2])
                    sbj_boxes = tf.gather(bbox_ref, tuples[:,0])
                    obj_boxes = tf.gather(bbox_ref, tuples[:,2])

                    triplets = tf.stack([sbj_class, tuples[:,1], obj_class], 1)
                    boxes = tf.concat([sbj_boxes, obj_boxes], 1)
                    return triplets, boxes

                gt_tuples = tf.stack([rel_ref_gt[:,0], rel_class_gt, rel_ref_gt[:,1]], 1)
                gt_triplets, gt_boxes = get_triplets(obj_cat_gt, obj_bbox_gt, gt_tuples)
                pred_triplets, pred_boxes = get_triplets(obj_cat_pred, obj_bbox_pred, tmp_tuples)

                def evaluate_rels():
                    _, matched_rels = calc.tf_relation_recall(gt_triplets, pred_triplets,
                                                              gt_boxes, pred_boxes)
                    gt_rels_found = tf.reduce_max(matched_rels, 1)
                    pred_rel_TP = tf.reduce_max(matched_rels, 0)

                    # Check for missed detections
                    missed_gt_idxs = tf.squeeze(tf.where(tf.equal(gt_rels_found, 0)),1)
                    missed_tuples = tf.gather(gt_tuples, missed_gt_idxs)
                    # Remap gt obj idxs to pred obj idxs
                    remap_sbjs = tf.gather(pred_idx_ref, missed_tuples[:,0])
                    remap_objs = tf.gather(pred_idx_ref, missed_tuples[:,2])
                    missed_tuples = tf.stack([remap_sbjs, missed_tuples[:,1], remap_objs],1)

                    n_missed = tf.shape(missed_gt_idxs)[0]
                    to_concat = [tf.zeros([n_missed,1]), missed_tuples, 2*tf.ones([n_missed,1])]
                    missed_rels = tf.concat([tf.to_float(v) for v in to_concat], 1)

                    return gt_rels_found, pred_rel_TP, missed_rels

                def empty_vals():
                    return tf.zeros([],tf.int32), tf.zeros([tf.shape(tmp_tuples)[0]],tf.int32), tf.zeros([0,5])

                gt_rels_found, pred_rel_TP, missed_rels = tf.cond(n_gt_rels > 0,
                                                                  evaluate_rels, empty_vals)

                # Prepare relationship reference array
                to_concat = [tf.expand_dims(tmp_scores,1), tmp_tuples, tf.expand_dims(pred_rel_TP,1)]
                rels = tf.concat([tf.to_float(v) for v in to_concat], 1)
                rels = tf.concat([rels, missed_rels], 0)
                final_n_rels = tf.shape(tmp_tuples)[0]

                return rels, final_n_rels, gt_rels_found

            rels, n_rels, gt_rels_found = tf.cond(tf.minimum(n_rels, n_objs) > 0, postprocess_rels,
                                                  lambda: (tf.zeros([0,5]), tf.zeros([],tf.int32), tf.zeros([],tf.int32)))

            # Pad objs and rels so each sample returns same shape tensor
            objs = tf.pad(objs, [[0,2*opt.max_nodes-tf.shape(objs)[0]],[0,0]], constant_values=-1)
            rels = tf.pad(rels, [[0,2*opt.max_nodes-tf.shape(rels)[0]],[0,0]], constant_values=-1)

            all_objs += [objs]
            n_found_objs += tf.reduce_sum(gt_objs_found)
            obj_count += n_gt_objs

            all_rels += [rels]
            n_found_rels += tf.reduce_sum(gt_rels_found)
            rel_count += n_gt_rels

            pred_obj_count += [n_objs]
            pred_rel_count += [n_rels]

        all_objs = tf.stack(all_objs, 0)
        all_objs.set_shape([batchsize, 2*opt.max_nodes, 16 + opt.tag_dim])
        n_objs = tf.stack(pred_obj_count, 0)

        all_rels = tf.stack(all_rels, 0)
        all_rels.set_shape([batchsize, 2*opt.max_nodes, 5])
        n_rels = tf.stack(pred_rel_count, 0)

        obj_recall = tf.cond(obj_count >= 1, lambda: tf.to_float(n_found_objs / obj_count), lambda: tf.zeros([]))
        rel_recall = tf.cond(rel_count >= 1, lambda: tf.to_float(n_found_rels / rel_count), lambda: tf.zeros([]))

        pred = {'objs':all_objs, 'rels':all_rels, 'n_objs':n_objs, 'n_rels':n_rels}
        acc = {'obj_recall':obj_recall, 'rel_R@100':rel_recall}
        return pred, acc

    def get_summary_img(self, inp_img, objs, rels):
        """ Generate an image summarizing predicted vs ground truth graph

        Draws bounding boxes around objects, and lists tuples of all predicates
        while providing other information like the score associated with
        predictions and which ground truth predicates were missed.

        Args:
            inp_img: Input image.
            objs: Array with all object info (n x 24). Each row contains score,
                top-k classes and class scores, bounding box coordinates, tag,
                and a label indicating whether it is TP/FP/FN.
            rels: Array with all relationship info (n x 5). Each row contains
                score, subject index, class, object index, and label indicating
                whether it is TP/FP/FN.

        """
        
        tmp_img = scipy.misc.imresize(inp_img, (512,512))
        o_img = np.ones((512,430,3), np.uint8)*255
        r_img = np.ones((512,1024,3), np.uint8)*255
        sep = np.ones((512,1,3), np.uint8) * 0

        try:
            objs = objs[objs[:,1] != -1]
            rels = rels[rels[:,1] != -1]

            # Ref colors (0:FP = red, 1:TP = green, 2:FN = gray)
            color_ref = [(155,0,0),(0,155,0),(75,75,75)]
            row_height = 11
            start_pt = np.array([10,16])

            # Go through objects, drawing bounding boxes
            class_counts = {}
            obj_names = []
            o_img = img.draw_text(o_img, np.array([5,2]), 'Objects', (0,0,0), draw_bg=False)
            o_img = img.draw_text(o_img, start_pt, '{0:4}{1:6}{2:16}{3:5}'.format("Idx","Score","Class","Tag"), (0,0,0), draw_bg=False)

            for i in range(objs.shape[0]):
                obj_score = objs[i][0]
                obj_class = int(objs[i][1])
                obj_bbox = objs[i][11:15]
                obj_tag = objs[i][16]
                obj_label = int(objs[i][-1])

                obj_name = self.ds.class_labels[obj_class]
                if not obj_name in class_counts:
                    class_counts[obj_name] = 1
                else:
                    class_counts[obj_name] += 1
                obj_name = obj_name + " %d" % class_counts[obj_name]
                obj_names += [obj_name]
                # Transform box coordinates to scale them up
                tmp_bbox = (obj_bbox + .5) * 512 / self.opt.input_res
                # Draw class label and bounding box
                tmp_img = img.draw_bbox(tmp_img, tmp_bbox, color_ref[obj_label], 5)
                tmp_img = img.draw_text(tmp_img, tmp_bbox[:2], obj_name, color_ref[obj_label])

                # Write down object info in object column
                tmp_str = '{0:<4}{1:<6.2f}{2:16}{3:5.3f}'.format(i,obj_score, obj_name, obj_tag)
                if i < 42: offset = np.array([0, (i+1)*row_height])
                else: offset = np.array([210, (i-41)*row_height])
                tmp_pt = start_pt + offset
                o_img = img.draw_text(o_img, tmp_pt, tmp_str, color_ref[obj_label], draw_bg=False)

            # Prepare relationship tuple image
            r_img = img.draw_text(r_img, np.array([5,2]), 'Relationships', (0,0,0), draw_bg=False)
            r_img = img.draw_text(r_img, start_pt, '{0:6}{1:50}{2:12}{3:12}'.format("Score","Phrase","Subj","Obj"), (0,0,0), draw_bg=False)

            for i in range(rels.shape[0]):
                rel_score, rel_class = rels[i][0], int(rels[i][2])
                sidx, oidx = int(rels[i][1]), int(rels[i][3])
                rel_label = int(rels[i][-1])

                if sidx < len(obj_names) and oidx < len(obj_names) and rel_class < len(self.ds.relationships): 
                    p_str = "%s %s %s" % (obj_names[sidx], self.ds.relationships[rel_class], obj_names[oidx])
                    s_str = '{0:<4}'.format(sidx)
                    o_str = '{0:<4}'.format(oidx)

                    full_str = '{0:<6.2f}{1:50}{2:12}{3:12}'.format(rel_score, p_str, s_str, o_str)
                else:
                    full_str = 'oops, bad subject or object index'

                if i < 42: offset = np.array([0, (i+1)*row_height])
                else: offset = np.array([512, (i-41)*row_height])
                tmp_pt = start_pt + offset
                r_img = img.draw_text(r_img, tmp_pt, full_str, color_ref[rel_label], draw_bg=False)

        except:
            print("Error generating summary image.")

        return np.concatenate((tmp_img, sep, o_img, sep, r_img),1).astype(np.float32)

    def visualize_sample(self, idx, augment=False):
        """ Visualize ground truth annotations for a particular sample """

        sample = self.load_sample_data([idx, int(augment)])
        idx_ref = self.idx_ref

        # Objects (score, class, bbox, tag, label)
        obj_ref = sample[idx_ref['obj_ref']]
        n_objs = (obj_ref[:,0] > -1).sum()

        objs = -np.ones((self.opt.max_nodes, 24))
        objs[:n_objs,0] = 1             # score
        objs[:,1] = obj_ref[:,2]        # class
        objs[:,11:15] = obj_ref[:,3:]   # bbox
        objs[:,-1] = 2

        # Relationships (score, class, subject/object index, tags, label)
        rel_classes = sample[idx_ref['node_classes']][:,self.opt.obj_slots:]
        rel_ref = sample[idx_ref['rel_ref']]
        rel_tuples = rel_ref[rel_classes > -1]
        n_rels = (rel_classes > -1).sum()

        rels = -np.ones((self.opt.max_nodes, 7))
        rels[:n_rels,0] = 1                                 # score
        rels[:n_rels,1] = rel_tuples[:,0]                   # sbj idx
        rels[:n_rels,2] = rel_classes[rel_classes > -1]     # class
        rels[:n_rels,3] = rel_tuples[:,1]                   # obj idx
        rels[:,-1] = 2

        summ_img = self.get_summary_img(sample[idx_ref['inp_img']], objs, rels)

        return summ_img

    def setup_summaries(self, net, inp, label, loss, pred, accuracy):
        summaries, image_summaries = super(Task, self).setup_summaries(net, inp, label, loss, pred, accuracy)

        summary_img = tf.py_func(self.get_summary_img, [inp[0][0], pred['objs'][0], pred['rels'][0]], tf.float32)
        summary_img = tf.expand_dims(summary_img, 0)
        
        for s in ['train', 'valid']:
            tmp_image_summaries = [
                image_summaries[s],
                tf.summary.image(s + '_obj_hm', net[0][:,:,:,:1]),
                tf.summary.image(s + '_rel_hm', net[0][:,:,:,1:]),
                tf.summary.image(s + '_summary_img', summary_img),
            ]

            image_summaries[s] = tf.summary.merge(tmp_image_summaries)

        return summaries, image_summaries
