import tensorflow as tf
slim = tf.contrib.slim

from tensorflow.python.ops import control_flow_ops
import time
import numpy as np
import scipy.io as sio
import cv2, glob, os, re
import tracker_util as tutil


params = tutil.get_params()

data_home = 'OTB100'
dataset = glob.glob(os.path.join('%s'%data_home,'*'))
dataset.sort()
dataset = dataset[:100]

param_path = 'ADNet_params.mat'
results = {}
results['fps'] = []
results['location'] = []
results['overlap'] = []


nodes = {}
epoch = 0
initial_params = sio.loadmat(param_path)
with tf.Graph().as_default():
#%% get convolutional feature
    nodes['full_training'] = tf.placeholder_with_default(0.0, [])
    nodes['cropped_img'] = tf.placeholder_with_default(tf.cast(np.zeros([1,112,112,3]),tf.float32), [None, 112,112,3])
    nodes['cropped'] = tf.placeholder_with_default(1.0, [])
    nodes['boxes'] = tf.placeholder_with_default(tf.cast(np.zeros([1,4]),tf.float32), [None, 4])
    nodes['boxes_ind'] = tf.placeholder_with_default(tf.cast(np.zeros([1]),tf.int32), [None])
    nodes['image'] = tf.placeholder_with_default(tf.cast(np.zeros([1,112,112,3]),tf.float32), [None, None,None,3])
    nodes['crop'] = tf.image.crop_and_resize(nodes['image'], nodes['boxes'],  nodes['boxes_ind'], [112, 112])
    img = tf.cond(tf.equal(nodes['cropped'] ,0.0),
                  lambda : nodes['cropped_img'],
                  lambda : nodes['crop'])
    nodes['conv_feat'] = tutil.model_conv(img, param_path)
    
#%% get action and confidence
    nodes['conv'] = tf.placeholder_with_default(tf.cast(np.zeros([1,3,3,512]),tf.float32), [None, 3,3,512])
    input_feature = tf.cond(tf.equal(nodes['full_training'] ,1.0),
                            lambda : nodes['conv_feat'],
                            lambda : nodes['conv'])
    nodes['act_label'] = tf.placeholder_with_default(tf.cast(np.zeros([1,11]),tf.float32), [None, 11])
    nodes['sco_label'] = tf.placeholder_with_default(tf.cast(np.zeros([1,2]),tf.float32), [None, 2])
    nodes['action_hist'] = tf.placeholder_with_default(tf.cast(np.zeros([1,110]),tf.float32), [None, 110])
    nodes['is_training'] = tf.placeholder_with_default(0.0, [])
    train = tf.equal(nodes['is_training'],1.0)
    
    nodes['action'], conf, fc2 = tutil.model_fc(input_feature, nodes['action_hist'], train, param_path)
    nodes['soft_conf'] = tf.nn.softmax(conf)
    
#%% compute loss and train model
    nodes['act_loss'] = tf.losses.softmax_cross_entropy(nodes['act_label'], nodes['action'])
    nodes['sco_loss'] = tf.losses.softmax_cross_entropy(nodes['sco_label'], conf)
    
    learning_rate = tf.placeholder(tf.float32)
    
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    fc_tensor = [initial_params['fc1w'],initial_params['fc1b'][0],
                 initial_params['fc2w'],initial_params['fc2b'][0],
                 np.squeeze(initial_params['fc3w'].astype(np.float32)),
                 np.squeeze(initial_params['fc3b'].astype(np.float32)),
                 np.squeeze(initial_params['fc4w'].astype(np.float32)),
                 np.squeeze(initial_params['fc4b'].astype(np.float32)),]
    def assign(variables, fc_tensor):
        for i in range(6,14):
            variables[i].assign(fc_tensor[i-6])
        return variables
    
    assign_fc = tf.placeholder_with_default(0.0, [])
    v_ = tf.cond(tf.equal(assign_fc,1.0), lambda : fc_tensor,
                                          lambda : variables[6:14])
    fc_variables = []
    for i in range(6,14):
        fc_variables.append(variables[i].assign(v_[i-6]))
    
    nodes['act'] = tf.placeholder(tf.float32)
    losses = tf.cond(tf.equal(nodes['act'],1.0), lambda :   nodes['act_loss']+0*nodes['sco_loss'],
                                                 lambda : 0*nodes['act_loss']+  nodes['sco_loss'])
#    
    opimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = opimizer.compute_gradients(losses, var_list = variables[6:])
    for i in range(len(gradients)):
        if i%2 == 0:
            gradients[i] = (gradients[i][0]*10,gradients[i][1])
        else:
            gradients[i] = (gradients[i][0]*20,gradients[i][1])
    gradients[4] = (gradients[4][0]*2, gradients[4][1])
    gradients[5] = (gradients[5][0]*2, gradients[5][1])
        
    update_ops =[opimizer.apply_gradients(gradients, global_step=global_step)]
    update_op = tf.group(*update_ops)
    nodes['act_train_op'] = control_flow_ops.with_dependencies([update_op], losses, name='train_op')
    
    variables  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    reset = tf.placeholder_with_default(0.0, [])
    def do_nothing(variables):
        #do_nothing
        return variables
    def reset_momentum(variables):
        variables[15] = tf.assign(variables[15], tf.zeros_like(variables[15]))
        variables[16] = tf.assign(variables[16], tf.cast(0.9,dtype=tf.float32))
        variables[17] = tf.assign(variables[17], tf.cast(0.999,dtype=tf.float32))
        for i, v in enumerate(variables[18:]):
            variables[18+i] = tf.assign(v, tf.zeros_like(v))
        return variables
    variables = tf.cond(tf.equal(reset,1.0), lambda : reset_momentum(variables),
                                             lambda : do_nothing(variables))
    nodes['reset'] = reset
    nodes['learning_rate'] = learning_rate
#%% restore params and hardware setting
#    checkpoint_path = latest_checkpoint(train_dir)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for test_num in range(1):
            results = {}
            results['fps'] = []
            results['location'] = []
            results['overlap'] = []
            temp = []
            for v, vd in enumerate(dataset[0:]):
                sess.run(fc_variables, feed_dict = {assign_fc : 1.0})
                l = i+1
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
                        
                        pos_examples = tutil.gen_samples('gaussian', gt, params['pos_init']*2,
                                                         params, params['finetune_trans'], params['finetune_scale_factor'])
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
                        feat_conv = tutil.get_conv_feature(sess, nodes['conv_feat'], 
                                                           feed_dict = {nodes['cropped'] : 1.0, nodes['boxes_ind'] : np.array([0]*examples.shape[0]),
                                                                        nodes['image'] : [img], nodes['boxes'] : tutil.refine_box(examples, params)})
                        pos_data = feat_conv[:pos_examples.shape[0]]
                        neg_data = feat_conv[pos_examples.shape[0]:]
                        
                        pos_action_labels = tutil.gen_action_labels(params, pos_examples, gt)
                        
                        _ = sess.run(variables, feed_dict = {reset : 1.0})
                        tutil.train_fc(sess, nodes, feat_conv, pos_action_labels,
                                       params['iter_init'], params, params['init_learning_rate'])
    
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
                        
                        t = time.time()
                        total_moves = 0
                        move_counter = 0
                        check = 0
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
                            
                            pred, pred_score = sess.run([nodes['action'], nodes['soft_conf']],
                                                        feed_dict = {nodes['image'] : [img], nodes['cropped'] : 1.0,
                                                                     nodes['full_training'] : 1.0, nodes['boxes_ind'] : np.array([0]), 
                                                                     nodes['boxes'] : tutil.refine_box(np.expand_dims(curr_bbox,0), params),
                                                                     nodes['action_hist'] : action_history_oh.reshape(1,-1)})
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
                            red_score_pred = sess.run(nodes['soft_conf'],
                                                  feed_dict = {nodes['image']: [img], nodes['cropped'] : 1.0,
                                                               nodes['full_training'] : 1.0, nodes['boxes_ind'] : np.array([0]*samples_redet.shape[0]),
                                                               nodes['boxes'] : tutil.refine_box(samples_redet, params),
                                                               nodes['action_hist'] : np.vstack([action_history_oh.reshape(1,-1)]*samples_redet.shape[0]),
                                                               nodes['is_training'] : 0.0})
        
        
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
                            feat_conv = tutil.get_conv_feature(sess, nodes['conv_feat'],
                                                               feed_dict = {nodes['cropped'] : 1.0, nodes['boxes_ind'] : np.array([0]*examples.shape[0]),
                                                                            nodes['image'] : [img], nodes['boxes'] : tutil.refine_box(examples, params)})
            
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
#                            if check == 5:
                            _ = sess.run(variables, feed_dict = {reset : 1.0})
#                                check = 0
                            iteration = params['iter_on']
#                            if is_negative:
#                                iteration = params['iter_on']//2
                            tutil.train_fc(sess, nodes, feat_conv, pos_action_labels,
                                           iteration, params, params['on_learning_rate'])
                            check += 1
                            
                    full_history.append(curr_bbox)
                    full_gt.append(gt)
                    total_moves += move_counter
                    
                    frame = np.copy(img)
                    frame = cv2.rectangle(frame,(int(gt[0]),int(gt[1])),
                                                (int(gt[0]+gt[2]),int(gt[1]+gt[3])),[0,0,255],2)
                    frame = cv2.rectangle(frame,(int(curr_bbox[0]),int(curr_bbox[1])),
                                                (int(curr_bbox[0]+curr_bbox[2]),int(curr_bbox[1]+curr_bbox[3])),[255,0,0],2)
                    
                    cv2.imwrite('results/'+frames[f][-8:],frame)
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
                
                print ('Vid : %s Fps : %2f location accuracy : %5f overlap accuracy : %5f'%(str(v).rjust(2, '0'), np.mean(results['fps']), np.mean(np.array(results['location'])[:,20]),np.mean(results['overlap'])))
                key = cv2.waitKey(100) & 0xff
                if key == ord('q'):
                    break
                        
sess.close()
cv2.destroyAllWindows()
