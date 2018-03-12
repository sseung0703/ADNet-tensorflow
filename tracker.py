import tensorflow as tf
slim = tf.contrib.slim
#from nets import nets_factory
#from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.python.training.saver import Saver
from tensorflow.python.training       import supervisor
from tensorflow import Session
from tensorflow import ConfigProto

import time
import numpy as np

import scipy.io as sio
import cv2, glob, os, re

import tracker_util as tutil
params = tutil.get_params()
data_home = '/home/dmsl/Documents/data/OTB100/'
param_path = '/home/dmsl/Documents/Adnet-tf'
dataset = glob.glob(os.path.join('%s'%data_home,'*'))
dataset.sort()

#results = {}
#results['fps'] = []
#results['location'] = []
#results['overlap'] = []
for v, vd in enumerate(dataset):
#    vd = dataset[v+52]
    with tf.Graph().as_default():
    #%% get convolutional feature
        cropped_img = tf.placeholder_with_default(tf.cast(np.zeros([1,112,112,3]),tf.float32), [None, 112,112,3])
        cropped = tf.placeholder_with_default(1.0, [])
        boxes = tf.placeholder_with_default(tf.cast(np.zeros([1,4]),tf.float32), [None, 4])
        boxes_ind = tf.placeholder_with_default(tf.cast(np.zeros([1]),tf.int32), [None])
        image = tf.placeholder_with_default(tf.cast(np.zeros([1,112,112,3]),tf.float32), [None, None,None,3])
        crop = tf.image.crop_and_resize(image, boxes,  boxes_ind, [112, 112])
        img = tf.cond(tf.equal(cropped ,0.0),
                      lambda : cropped_img,
                      lambda : crop)
        conv_feat = tutil.model_conv(img, param_path)
        
    #%% get action and confidence
        conv = tf.placeholder_with_default(tf.cast(np.zeros([1,3,3,512]),tf.float32), [None, 3,3,512])
        act_label = tf.placeholder_with_default(tf.cast(np.zeros([1,11]),tf.float32), [None, 11])
        sco_label = tf.placeholder_with_default(tf.cast(np.zeros([1,2]),tf.float32), [None, 2])
        action_hist = tf.placeholder_with_default(tf.cast(np.zeros([1,110]),tf.float32), [None, 110])
        is_training = tf.placeholder_with_default(0.0, [])
        train = tf.equal(is_training,1.0)
        
        action, conf, fc2 = tutil.model_fc(conv, action_hist, train, param_path)
        soft_conf = tf.nn.softmax(conf)
        
    #%% compute loss & accuracy and train model
        mask = tf.slice(sco_label,[0,1],[-1,1])
        num_act_img = tf.reduce_sum(mask)
        act = action*mask
        
        act_loss = tf.losses.softmax_cross_entropy(act_label, act, reduction = tf.losses.Reduction.NONE)
        act_loss = tf.reduce_sum(tf.expand_dims(act_loss,1)*mask)/num_act_img
        
        sco_loss = tf.losses.softmax_cross_entropy(sco_label,conf)
        
        action_train = tf.placeholder_with_default(0.0, [])
        
        learning_rate = tf.placeholder(tf.float32)
#        act_train_op = tf.train.AdamOptimizer(learning_rate).minimize(act_loss)
#        sco_train_op = tf.train.AdamOptimizer(learning_rate).minimize(sco_loss)
        act_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(act_loss)
        sco_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(sco_loss)
        
    #%% restore params and hardware setting
    #    checkpoint_path = latest_checkpoint(train_dir)
        config = ConfigProto()
        config.gpu_options.allow_growth=True
        correct = 0
        predict = 0
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            ground_truth = open('%s/groundtruth_rect.txt'%vd)
            frames = glob.glob(os.path.join('%s/img'%vd, '*.jpg'))
            frames.sort()
            
            total_pos_data = {}
            total_neg_data = {}
            total_pos_action_labels = {}
            total_pos_examples = {}
            total_neg_examples = {}
            
            frame_window = []
            
            cont_negatives = 0
            
            full_history = []
            full_gt = []
            for f, frame in enumerate(frames):
                gt = ground_truth.readline()
                gt = np.array(re.findall('\d+',gt),dtype=int)
                img = cv2.imread(frame)
                params['height'], params['width'] = img.shape[:2]
                
#%% Init Frame
                if f == 0:
                    action_history_oh = np.zeros([params['num_action_history'],params['num_actions']]);
                    
                    pos_examples = tutil.gen_samples('gaussian', gt, params['pos_init']*2, params, params['finetune_trans'], params['finetune_scale_factor'])
                    r = tutil.overlap_ratio(pos_examples,gt)
                    pos_examples = pos_examples[np.where(r>params['pos_thr_init'])]
                    pos_examples = pos_examples[np.random.choice(pos_examples.shape[0],
                                                                 min(params['pos_init'], pos_examples.shape[0]),replace=False)]
    
                    neg_examples = np.vstack((tutil.gen_samples('uniform', gt, params['neg_init'] , params, 1, 10),
                                             tutil.gen_samples('whole', gt, params['neg_init'] , params, 1, 10)))
                    r = tutil.overlap_ratio(neg_examples,gt)
                    neg_examples = neg_examples[np.where(r<params['neg_thr_init'])]
                    neg_examples = neg_examples[np.random.choice(neg_examples.shape[0],
                                                                 min(params['neg_init'], neg_examples.shape[0]),replace=False)]
                    examples = np.vstack((pos_examples, neg_examples))
                    feat_conv = tutil.get_conv_feature(sess, conv_feat, 
                                                       feed_dict = {cropped : 1.0, boxes_ind : np.array([0]*examples.shape[0]),
                                                                    image : [img], boxes : tutil.refine_box(examples, params)})
                    pos_data = feat_conv[:pos_examples.shape[0]]
                    neg_data = feat_conv[pos_examples.shape[0]:]
                    
                    pos_action_labels = tutil.gen_action_labels(params, pos_examples, gt)
                    
                    tutil.train_fc(sess, [act_train_op, sco_train_op, soft_conf], params['iter_init'],
                                   conv, act_label,sco_label,is_training, action_train, learning_rate,
                                   params,feat_conv, pos_action_labels, params['init_learning_rate'])
                    
                    
                    total_pos_data['%d'%f] = pos_data
                    total_neg_data['%d'%f] = neg_data
                    total_pos_action_labels['%d'%f] = pos_action_labels
                    total_pos_examples['%d'%f] = pos_examples
                    total_neg_examples['%d'%f] = neg_examples
                    
                    frame_window.append(f)
                    is_negative = False
                    
                    action_history = np.zeros([params['num_show_actions']]);
                    this_actions = np.zeros([params['num_show_actions'],11]);
                    
                    curr_bbox = gt.astype(np.float32)
                    
                    pre = [0,0,0]
                    
                    t = time.time()
                    total_moves = 0
                    move_counter = 0
#%% do tracking
                else:
                    
                    curr_bbox_old = curr_bbox
                    move_counter = 0
                    target_score = 0
                    
                    num_action_step_max = 20;
                    bb_step = np.zeros([num_action_step_max, 4])
                    score_step = np.zeros([num_action_step_max, 1])
                    is_negative = False
                    prev_score = -9999
                    this_actions = np.zeros([params['num_show_actions'],1])
                    action_history_oh_old = action_history_oh
                    
                    while (move_counter < num_action_step_max):
                        bb_step[move_counter] = curr_bbox
                        score_step[move_counter] = prev_score;
                        
                        action_history_oh *= 0

                        for i, act in enumerate(action_history[:params['num_action_history']]):
                            if act<11:
                                action_history_oh[i,int(act)] = 1
                        
                        curr_feat_conv, crop_img = tutil.get_conv_feature(sess, [conv_feat,crop],
                                                                          feed_dict = {cropped : 1.0, boxes_ind : np.array([0]),
                                                                                       image : [img], boxes : tutil.refine_box(np.expand_dims(curr_bbox,0), params)})
        
                        pred, pred_score, fc = sess.run([action, soft_conf, fc2],
                                                        feed_dict = {conv : curr_feat_conv,
                                                                     action_hist : action_history_oh.reshape(1,-1),
                                                                     is_training : 0.0})
                        curr_score = pred_score[0,1]
                        max_action = np.argmax(pred[0])
                        if (curr_score < params['failedThre']):
                            is_negative = True;
                            curr_score = prev_score;
                            action_history[1:] = action_history[:-1]
                            action_history[0] = 12;
                            cont_negatives += 1;
                            break;
                            
                        curr_bbox = tutil.do_action(curr_bbox, max_action, params);
                        
                        if ((len(np.where(np.sum(np.equal(np.round(bb_step),np.round(curr_bbox)),1)==4)[0]) > 0)
                           & (max_action != params['stop_action'])):
                            max_action = params['stop_action']
                        
                        
                        action_history[1:] = action_history[:-1]
                        action_history[0] = max_action;
                        target_score = curr_score;
                        
                        if max_action == params['stop_action']:        
                            break
                        
                        move_counter += 1;
                        prev_score = curr_score;
                        
#%% Tracking Fail --> Re-detection  
                    if ((f > 0) & (is_negative == True)):
#                        print (f)
#                        cv2.waitKey(0)
                        total_pos_data['%d'%f] = np.zeros([0,3,3,512])
                        total_neg_data['%d'%f] = np.zeros([0,3,3,512])
                        total_pos_action_labels['%d'%f] = np.zeros([0,11])
                        total_pos_examples['%d'%f] = np.zeros([0,4])
                        total_neg_examples['%d'%f] = np.zeros([0,4])
                        
                        samples_redet = tutil.gen_samples('gaussian', curr_bbox_old, params['redet_samples'], params, min(1.5, 0.6*1.15**cont_negatives), params['finetune_scale_factor'])
                        feat_conv = tutil.get_conv_feature(sess, conv_feat,
                                                           feed_dict = {cropped : 1.0, boxes_ind : np.array([0]*samples_redet.shape[0]),
                                                                        image : [img], boxes : tutil.refine_box(samples_redet, params)})
    
                        red_score_pred = sess.run(conf,
                                              feed_dict = {conv : feat_conv,
                                              action_hist : np.vstack([action_history_oh.reshape(1,-1)]*feat_conv.shape[0]),
                                              is_training : 0.0})
    
    
                        idx = np.lexsort((np.array(range(params['redet_samples'])),red_score_pred[:,1]))
                        target_score = np.mean(red_score_pred[(idx[-5:]),1])
                        if target_score > curr_score:
                            curr_bbox = np.mean(samples_redet[(idx[-5:]),:],0)
                        move_counter += params['redet_samples']

#%% Tracking Success --> generate samples
                    if ((f > 0) & ((is_negative == False) | (target_score > params['successThre']))):
                        cont_negatives = 0;
                        pos_examples = tutil.gen_samples('gaussian', curr_bbox, params['pos_on']*2, params, params['finetune_trans'], params['finetune_scale_factor'])
                        r = tutil.overlap_ratio(pos_examples,curr_bbox)
                        pos_examples = pos_examples[np.where(r>params['pos_thr_on'])]
                        pos_examples = pos_examples[np.random.choice(pos_examples.shape[0],
                                                                     min(params['pos_on'], pos_examples.shape[0]),replace=False)]
                        
                        neg_examples = tutil.gen_samples('uniform', curr_bbox, params['neg_on']*2, params, 2, 5)
                        r = tutil.overlap_ratio(neg_examples,curr_bbox)
                        neg_examples = neg_examples[np.where(r<params['neg_thr_on'])]
                        neg_examples = neg_examples[np.random.choice(neg_examples.shape[0],
                                                                     min(params['neg_on'], neg_examples.shape[0]),replace=False)]
                        
                        examples = np.vstack((pos_examples, neg_examples))
                        feat_conv = tutil.get_conv_feature(sess, conv_feat,
                                                           feed_dict = {cropped : 1.0, boxes_ind : np.array([0]*examples.shape[0]),
                                                                        image : [img], boxes : tutil.refine_box(examples, params)})
        
                        total_pos_data['%d'%f] = feat_conv[:pos_examples.shape[0]]
                        total_neg_data['%d'%f] = feat_conv[pos_examples.shape[0]:]
                        
                        pos_action_labels = tutil.gen_action_labels(params, pos_examples, curr_bbox)
                        
                        total_pos_action_labels['%d'%f] = pos_action_labels
                        total_pos_examples['%d'%f] = pos_examples
                        total_neg_examples['%d'%f] = neg_examples
        
                        frame_window.append(f)
                        
                        if (len(frame_window) > params['frame_long']):
                            total_pos_data['%d'%frame_window[-params['frame_long']]] = np.zeros([0,3,3,512])
                            total_pos_action_labels['%d'%frame_window[-params['frame_long']]] = np.zeros([0,11])
                            total_pos_examples['%d'%frame_window[-params['frame_long']]] = np.zeros([0,4])
                            
                        if (len(frame_window) > params['frame_short']):
                            total_neg_data['%d'%frame_window[-params['frame_short']]] = np.zeros([0,3,3,512])
                            total_neg_examples['%d'%frame_window[-params['frame_short']]] = np.zeros([0,4])
                            
#%% Do online-training
                    if ( ((f+1)%params['iterval'] == 0)  | (is_negative == True) ):
                        if (f+1)%params['iterval'] == 0:
                            f_st = max(0,len(frame_window)-params['frame_long'])
                            
                            pos_data = []
                            pos_action_labels = []
                            for wind in frame_window[f_st:]:
                                pos_data.append(total_pos_data['%d'%wind])
                                pos_action_labels.append(total_pos_action_labels['%d'%wind])
                                
                            pos_data = np.vstack(pos_data)
                            pos_action_labels = np.vstack(pos_action_labels)
                            
                        else:
                            f_st = max(0,len(frame_window)-params['frame_short'])
                            
                            pos_data = []
                            pos_action_labels = []
                            for wind in frame_window[f_st:]:
                                pos_data.append(total_pos_data['%d'%wind])
                                pos_action_labels.append(total_pos_action_labels['%d'%wind])
                                
                            pos_data = np.vstack(pos_data)
                            pos_action_labels = np.vstack(pos_action_labels)
                        
                        f_st = max(0,len(frame_window)-params['frame_short'])
                        neg_data = []
                        for wind in frame_window[f_st:]:
                            neg_data.append(total_neg_data['%d'%wind])
                            
                        neg_data = np.vstack(neg_data)
                        
                        feat_conv = np.vstack((pos_data, neg_data))
                        tutil.train_fc(sess, [act_train_op,sco_train_op,soft_conf], params['iter_on'],
                                       conv, act_label,sco_label,is_training, action_train, learning_rate,
                                       params,feat_conv, pos_action_labels, params['on_learning_rate'])
                            
                        
                full_history.append(curr_bbox)
                full_gt.append(gt)
                total_moves += move_counter
                
                frame = np.copy(img)
                frame = cv2.rectangle(frame,(int(gt[0]),int(gt[1])),
                                            (int(gt[0]+gt[2]),int(gt[1]+gt[3])),[0,0,255],2)
                frame = cv2.rectangle(frame,(int(curr_bbox[0]),int(curr_bbox[1])),
                                            (int(curr_bbox[0]+curr_bbox[2]),int(curr_bbox[1]+curr_bbox[3])),[255,0,0],2)
                cv2.imshow('f',frame)
                key = cv2.waitKey(1) & 0xff
                if key == ord('s'):
                    break
                
            total_time = time.time()-t
            fps = (f+1)/total_time
            full_history = np.array(full_history,dtype=float)
            full_gt = np.array(full_gt,dtype=float)
            results['fps'].append(fps)
            results['location'].append(tutil.location_precision(full_history, full_gt))
            results['overlap'].append(tutil.overlap_precision(full_history, full_gt))
            
            print ('Vid : %s Fps : %2f location accuracy : %5f overlap accuracy : %5f'%(str(v).rjust(2, '0'), np.mean(results['fps']), np.mean(np.mean(results['location'],1)),np.mean(np.mean(results['overlap'],1))))
            key = cv2.waitKey(100) & 0xff
            if key == ord('q'):
                break
                    
        sess.close()
cv2.destroyAllWindows()
