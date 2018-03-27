import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import scipy.io as sio
def get_params():
    params = {}
    
    params['iter_init'] = 20
    params['iter_on'] = 10
    params['iterval'] = 30
    params['init_learning_rate'] = 3e-4
    params['on_learning_rate'  ] = 1e-4
    params['batch_size'] = 64
    params['data_per_label'] = [6]*8+[4]+[6]*2
    params['redet_samples' ] = 64
    
    params['pos_init'] = 200
    params['neg_init'] = 150
    params['pos_on'] = 30
    params['neg_on'] = 15
    
    params['frame_long'] = 100
    params['frame_short'] = 20
    
    params['pos_thr_init'] = 0.7
    params['neg_thr_init'] = 0.3
    params['pos_thr_on'] = 0.7
    params['neg_thr_on'] = 0.5
    
    params['finetune_trans'] = 0.1
    params['scale_factor'] = 1.05
    params['finetune_scale_factor'] = 3
    
    params['num_actions'] = 11
    params['stop_action'] = 8
    params['deltas'] = np.array([[-1, 0, 0, 0],[-2, 0, 0, 0],
                                 [ 1, 0, 0, 0],[ 2, 0, 0, 0],
                                 [ 0,-1, 0, 0],[ 0,-2, 0, 0],
                                 [ 0, 1, 0, 0],[ 0, 2, 0, 0],
                                 [ 0, 0, 0, 0],
                                 [ 0, 0,-1,-1],[ 0 ,0, 1, 1]])
    params['stopIOU'] = 0.93
    
    params['successThre'] = 0.5
    params['failedThre'] = 0.5

    params['num_show_actions'] = 20
    params['num_action_history'] = 10
    
    return params

def gen_samples(noise, bb, n, params, tans_f, scale_f):
    w = params['width']; h = params['height']
    sample = [[bb[0]+bb[2]/2, bb[1]+bb[3]/2]+list(bb[2:])];
    samples = np.array(sample*n)
    
    if noise == 'gaussian':
        samples[:,:2] += tans_f * round(np.mean(bb[2:])) * np.maximum(-1,np.minimum(1,0.5*np.random.randn(n,2)))                                                                  
        samples[:,2:] *= np.hstack(tuple([params['scale_factor']**(scale_f*np.maximum(-1,np.minimum(1,0.5*np.random.randn(n,1))))]))
        
    elif noise == 'uniform':
        samples[:,:2] += tans_f * round(np.mean(bb[2:])) * (np.random.randn(n,2)*2-1)
        samples[:,2:] *= np.hstack(tuple([params['scale_factor']**(scale_f*np.random.randn(n,1)*2-1)]*2))
    elif noise == 'whole':
        ran = np.round([bb[2]/2, bb[3]/2, w-bb[2]/2, h-bb[3]/2]).astype(int)
        stride = np.round([bb[2]/5, bb[3]/5]).astype(int)
        dx, dy, ds = np.meshgrid(list(range(ran[0],ran[2],stride[0])),
                         list(range(ran[1],ran[3],stride[1])),
                         list(range(-5,5)),indexing='xy')
        
        windows = np.vstack((dx.reshape(-1),
                   dy.reshape(-1),
                   bb[2]*params['scale_factor']**ds.reshape(-1),
                   bb[3]*params['scale_factor']**ds.reshape(-1))).T
        
        samples = np.zeros((0,4));
        while (samples.shape[0]<n):
            samples = np.vstack((samples,windows[np.random.choice(windows.shape[0],min(windows.shape[0],n-samples.shape[0]),replace=False)]))
    
    samples[:,2] = np.maximum(10,np.minimum(w-10,samples[:,2]))
    samples[:,3] = np.maximum(10,np.minimum(h-10,samples[:,3]))
    
    samples[:,0] -= samples[:,2]/2
    samples[:,1] -= samples[:,3]/2
    samples[:,0] = np.maximum(1-samples[:,2]/2,np.minimum(w-samples[:,2]/2, samples[:,0]))
    samples[:,1] = np.maximum(1-samples[:,3]/2,np.minimum(h-samples[:,3]/2, samples[:,1]))
    samples = np.round(samples).astype(np.float32)
    return samples

def overlap_ratio(boxes1, refer):
    inter_boxes = [np.maximum(boxes1[:,0],refer[0]),
                   np.maximum(boxes1[:,1],refer[1]),
                   np.minimum(boxes1[:,0]+boxes1[:,2],refer[0]+refer[2]),
                   np.minimum(boxes1[:,1]+boxes1[:,3],refer[1]+refer[3])]
    
    inter_area = (inter_boxes[2]-inter_boxes[0])*(inter_boxes[3]-inter_boxes[1])
    union_area = boxes1[:,2]*boxes1[:,3]+refer[2]*refer[3]-inter_area
    
    return inter_area/union_area
    
def gen_action_labels(params, samples, bb):
    bb_samples = np.copy(samples)
    num = bb_samples.shape[0]
    
    bb_samples[:,0] += 0.5*bb_samples[:,2]
    bb_samples[:,1] += 0.5*bb_samples[:,3]
    
    deltas = np.maximum(np.hstack([bb_samples[:,2:]*0.03]+[bb_samples[:,2:]*0.03]),1)
    ar = bb_samples[:,2]/bb_samples[:,3]
    
    deltas[:,2] = np.where(deltas[:,0]<deltas[:,1], deltas[:,1]*ar,deltas[:,0])
    deltas[:,3] = np.where(deltas[:,0]>deltas[:,1], deltas[:,0]/ar,deltas[:,1])
    
    deltas = np.transpose(np.dstack([deltas]*params['num_actions']),[0,2,1])
    action_deltas = deltas*params['deltas']
    
    action_boxes = np.transpose(np.dstack([bb_samples]*params['num_actions']),[0,2,1])+ action_deltas
    action_boxes[:,:, 0] -= 0.5 * action_boxes[:,:, 2]
    action_boxes[:,:, 1] -= 0.5 * action_boxes[:,:, 3]
    
    overs = overlap_ratio(np.reshape(action_boxes,[-1,4]), bb)
    overs = np.reshape(overs,[-1,params['num_actions']])
    max_value = np.max(overs[:,:-2],1)
    max_action = np.argmax(overs[:,:-2],1)
    
    max_action = np.where(overs[:,params['stop_action']]>params['stopIOU'], params['stop_action'], max_action)
    max_action = np.where(max_value==overs[:,params['stop_action']], np.argmax(overs,1), max_action)
    
    action = np.zeros([num,params['num_actions']])
    action[tuple(range(num)), tuple(max_action)] = 1
    
    return action
def model_conv(image, path):
    vggm = sio.loadmat(path)
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(5e-4)):
        
        with tf.variable_scope('Adnet'):
            conv = slim.conv2d(image, 96, [7, 7], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                               weights_initializer = tf.constant_initializer(vggm['conv1w']),
                               biases_initializer  = tf.constant_initializer(vggm['conv1b']),
                               scope='conv0', trainable=True)
            conv = tf.nn.lrn(conv,2,2,1e-4,0.75)
            conv = slim.max_pool2d(conv, [3, 3], 2, scope='pool0')
            
            conv = slim.conv2d(conv, 256, [5, 5], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                               weights_initializer = tf.constant_initializer(vggm['conv2w']),
                               biases_initializer  = tf.constant_initializer(vggm['conv2b']),
                               scope='conv1', trainable=True)
            conv = tf.nn.lrn(conv,2,2,1e-4,0.75)
            conv = slim.max_pool2d(conv, [3, 3], 2, scope='pool1')
            
            conv = slim.conv2d(conv, 512, [3, 3], 1, padding = 'VALID', activation_fn=tf.nn.relu,
                               weights_initializer = tf.constant_initializer(vggm['conv3w']),
                               biases_initializer  = tf.constant_initializer(vggm['conv3b']),
                               scope='conv2', trainable=True)
                
    return conv

def model_fc(conv, action_hist, is_training, path):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(5e-4),
                        biases_regularizer=slim.l2_regularizer(5e-4),):
        with tf.variable_scope('Adnet'):
            fc = slim.conv2d(conv, 512, [3,3], padding='VALID', activation_fn=tf.nn.relu,
                             trainable=True, scope = 'full0')
            fc = slim.conv2d(fc, 512, [1,1], padding='VALID', activation_fn=None,
                             trainable=True, scope = 'full1')
            def add_hist(fc, action_hist):
                action_hist = tf.reshape(action_hist,[-1,1,1,110])
                fc += slim.conv2d(action_hist, 512, [1,1], activation_fn=None, 
                                  biases_initializer  = None,
                                  trainable=False, scope = 'fc_act')
                
                return tf.nn.relu(fc)
            
            def no_add_hist(fc):
                #do nothing
                return tf.nn.relu(fc)
            fc = tf.cond(is_training, lambda : no_add_hist(fc),
                                      lambda : add_hist(fc, action_hist))
            fc2 = slim.flatten(fc)
            act = slim.fully_connected(fc2, 11, activation_fn=None, 
                                       trainable=True, scope = 'full_act')
            conf = slim.fully_connected(fc2, 2, activation_fn=None, 
                                        trainable=True, scope = 'full_conf')
#            
        return act, conf, fc2

def refine_box(boxes, params):
    refined = np.copy(boxes)
    
    refined[:,:2] = np.maximum(0,refined[:,:2])
    refined[:,2] = np.minimum(params['width'],refined[:,2])
    refined[:,3] = np.minimum(params['height'],refined[:,3])
    
    refined[:,2] += refined[:,0]
    refined[:,3] += refined[:,1]
    refined = refined[:,(1,0,3,2)]
    refined[:,(0,2)] /= params['height']
    refined[:,(1,3)] /= params['width']
    
    return refined

def get_conv_feature(sess, end_points, feed_dict):
    conv_feat = sess.run(end_points, feed_dict = feed_dict)
    return conv_feat

def train_fc(sess, nodes, feat_conv, pos_action_labels,
             iteration, params, lr):
    num_data = feat_conv.shape[0]
    num_neg  = num_data-pos_action_labels.shape[0]
    num_pos  = num_data-num_neg
    
    pos_feat_conv = feat_conv[:num_pos]
    neg_feat_conv = feat_conv[num_pos:]
    
    randpos = np.random.choice(num_pos,num_pos,replace=False)
    randneg = np.random.choice(num_neg,num_neg,replace=False)
    
    train_action_cnt = 0
    train_action = []
    remain = params['batch_size']*iteration
    while remain > 0:
        if (train_action_cnt == 0):
            train_action_list = randpos
        
        train_action += list(train_action_list[train_action_cnt:min(len(train_action_list),train_action_cnt+remain)])
        train_action_cnt = min(len(train_action_list), train_action_cnt+remain)
        train_action_cnt = train_action_cnt%len(train_action_list)
        remain = params['batch_size']*iteration-len(train_action)
    
    train_neg_cnt = 0
    train_neg = []
    remain = 4*params['batch_size']*iteration
    while remain > 0:
        if train_neg_cnt == 0:
            train_hard_list = randneg

        train_neg += list(train_hard_list[train_neg_cnt:min(len(train_hard_list),train_neg_cnt+remain)])
        
        train_neg_cnt = min(len(train_hard_list),train_neg_cnt+remain)
        train_neg_cnt = train_neg_cnt%len(train_hard_list)
        remain = 4*params['batch_size']*iteration-len(train_neg)
    train_neg = np.array(train_neg)
    action_label_indices = []
    for i in range(11):
        action_label_indices.append(np.where(np.argmax(pos_action_labels,1)==i)[0])
        if len(action_label_indices[i]) == 0:
            action_label_indices[i] = np.random.choice(num_pos,num_pos, replace = False)
    
    train_actions = []
    for i in range(11):
        train_action_cnt = 0
        train_action = []
        
        remain = params['data_per_label'][i]*iteration
        action_label_index = action_label_indices[i]
        
        dpl = params['data_per_label'][i]
        ali = list(action_label_index)
        randpos = np.random.choice(ali,dpl)
        
        if np.size(randpos) == 0:
            randpos = np.random.choice(num_pos,num_pos, replace = False)
        
        while remain > 0:
            if(train_action_cnt==0):
                train_action_list = randpos
                
            train_action += list(train_action_list[train_action_cnt:min(len(train_action_list),train_action_cnt+remain)])
            train_action_cnt = min(len(train_action_list), train_action_cnt+remain)
            train_action_cnt = train_action_cnt%len(train_action_list)
            remain =params['data_per_label'][i]*iteration-len(train_action)
        train_actions.append(train_action)
    
    for i in range(iteration):
        pos_examples = []
        for a in range(11):
            train_action = train_actions[a]
            pos_examples += train_action[i*params['data_per_label'][a]: (i+1)*params['data_per_label'][a]]
            
        pos_feat = pos_feat_conv[pos_examples]
        action_labels = pos_action_labels[pos_examples]
        
        sess.run(nodes['act_train_op'],
                 feed_dict = {nodes['conv'] : pos_feat,
                              nodes['act_label'] : action_labels,
                              nodes['sco_label'] : np.zeros([action_labels.shape[0], 2]),
                              nodes['is_training'] : 1.0,
                              nodes['act'] : 1.0,
                              nodes['learning_rate'] : lr})
        
        batch_sco = []
        hard_start = i*4*params['batch_size']
        for b in range(4):
            b_st = b*params['batch_size']
            curr_batch = train_neg[hard_start+b_st:hard_start+b_st+params['batch_size']]
            curr_feat = neg_feat_conv[curr_batch]
            
            pred_score = sess.run(nodes['soft_conf'],
                                  feed_dict = {nodes['conv'] : curr_feat,
                                               nodes['is_training'] : 0.0})
            batch_sco.append(pred_score)
            
        batch_sco = np.vstack(batch_sco)
        
        idx = np.flip(np.lexsort((np.array(range(len(batch_sco))),batch_sco[:,1])),0)+hard_start
        hard_score_examples = train_neg[idx[:params['batch_size']]]
        feat_conv_samples = np.vstack((pos_feat,neg_feat_conv[hard_score_examples]))
        conf_label_samples = np.array([[0,1]]*params['batch_size']+[[1,0]]*params['batch_size'])
        
        sess.run(nodes['act_train_op'],
                 feed_dict = {nodes['conv'] : feat_conv_samples,
                              nodes['act_label'] : np.zeros([conf_label_samples.shape[0], 11]),
                              nodes['sco_label'] : conf_label_samples,
                              nodes['is_training'] : 1.0,
                              nodes['act'] : 0.0,
                              nodes['learning_rate'] : lr})
    
def do_action(curr_bbox, act, params):
    bbox = np.copy(curr_bbox)
    bbox[:2] += 0.5 * bbox[2:]
    
    deltas = np.maximum(np.hstack([bbox[2:]*0.03]*2),1)
    ar = bbox[2]/bbox[3]
    deltas[2] = np.where(deltas[0]<deltas[1], deltas[3]*ar,deltas[2])
    deltas[3] = np.where(deltas[0]>deltas[1], deltas[2]/ar,deltas[3])
    
    
    action_delta = params['deltas'][act,:]*deltas
    bbox_next = bbox + action_delta
    bbox_next[:2] -= 0.5 * bbox_next[2:]
    
    bbox_next[0] = min(max(bbox_next[0],0), params['width'] - bbox_next[2])
    bbox_next[1] = min(max(bbox_next[1],0), params['height'] - bbox_next[3])
    
    bbox_next[2] = max(10, min(params['width'], bbox_next[2]))
    bbox_next[3] = max(10, min(params['height'], bbox_next[3]))
    
    return bbox_next

def location_precision(esti, gt):
    esti = esti[:, [1,0]] + esti[:, (3,2)] / 2
    gt = gt[:, [1,0]] + gt[:, (3,2)] / 2

    max_threshold = 50
    precisions = np.zeros(max_threshold);
    if esti.shape[0] != gt.shape[0]:
        n = min(esti.shape[0], gt.shape[0])
        esti = esti[n:]
        gt = gt[n:]
    distances = np.sqrt((esti[:,0]-gt[:,0])**2 +(esti[:,1] - gt[:,1])**2)
    distances[np.isnan(distances)] = 0;

    for p in range(max_threshold):
        precisions[p] = len(np.where((distances+1) <= p)[0])/distances.shape[0]
        
    return precisions

def overlap_precision(esti, gt):
    max_threshold = 100
    precisions = np.zeros(max_threshold);
    if esti.shape[0] != gt.shape[0]:
        n = min(esti.shape[0], gt.shape[0])
        esti = esti[n:]
        gt = gt[n:]
    else:
        n = esti.shape[0]
        
    overlab = np.zeros([n,1]);
    for i in range(n):
        overlab[i] = overlap_ratio(np.expand_dims(esti[i],0),gt[i])
        
    overlab[np.isnan(overlab)] = 0;

    for p in range(max_threshold):
        precisions[p] = len(np.where(overlab >= (p+1)/100)[0])/overlab.shape[0]

    return precisions
        
        
        
