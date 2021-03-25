import tensorflow as tf

# the regularization part for the convolutional layers and dense layers.
l2_reg1 = tf.contrib.layers.l2_regularizer(5e-3)
l2_reg2 = tf.contrib.layers.l2_regularizer(0.0)
# the intializer for kenel and bias
kernel_initializer = tf.initializers.variance_scaling()
bias_initializer = tf.initializers.constant(value=0.1)


# normalize the input spectrogram
def normalize(specs):
    with tf.variable_scope('normalize'):
        max_vals = tf.reduce_max(specs, axis=(1, 2), keepdims=True)
        # [Batch_size, n_bins, time_steps, channels]
        specs = specs / (max_vals + 1e-8)
        specs = tf.transpose(specs, [0, 2, 1, 3])
    return specs


# the model part mentioned in our paper
def onset_and_frames(inputs, mode):
    inputs = normalize(inputs)
    with tf.variable_scope('pitch'):
        net1 = acoustic_model(inputs, mode)
        net1= lstm_layer(net1, 512, mode)
        logits1 = fc_layer(net1, 88)

    with tf.variable_scope('frame'):
        net2 = acoustic_model(inputs, mode)
        logits2 = fc_layer(net2, 88)
        logits = tf.concat([logits1, logits2], axis=2) # 
        net2 = lstm_layer(logits, 128, mode)
        logits2 = fc_layer(net2, 88)

    return logits1, logits2


def  acoustic_model(inputs, mode):

    with tf.variable_scope('conv_net'):
        # fisrt conv layer
        net = tf.layers.conv2d(inputs, 8, [3, 5], padding='valid', kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(net, momentum=0.9, epsilon=1e-7, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)

        # second conv layer
        net = tf.layers.conv2d(net, 16, [3, 5], padding='same', kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(net, momentum=0.9, epsilon=1e-7, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [1, 2], padding='same')

        # third conv layer
        net = tf.layers.conv2d(net, 32, [3, 5], padding='valid', kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(net, momentum=0.9, epsilon=1e-7, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)

        # fourth conv layer
        net = tf.layers.conv2d(net, 64, [3, 5], padding='valid', kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(net, momentum=0.9, epsilon=1e-7, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [1, 2], padding='same')

        # fifth conv layer
        net = tf.layers.conv2d(net, 128, [3, 5], padding='valid', kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(net, momentum=0.9, epsilon=1e-7, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [1, 2], padding='same')

        shapes = net.get_shape().as_list() # [B, time_step, spec_bins, channels]

        net = tf.reshape(net, [-1, shapes[1], shapes[2]*shapes[3]])
        net = tf.layers.dropout(net, 0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu, kernel_initializer=kernel_initializer, 
          bias_initializer=bias_initializer, kernel_regularizer=l2_reg2)
        net = tf.layers.dropout(net, 0.5, training=(mode == tf.estimator.ModeKeys.TRAIN)) # [batch_size, time_len, 1024] 
    
    return net #[batch_size, time_step, spec_bins, channel]


def lstm_layer(net, units, mode):
    # the bi lstm layer
    with tf.variable_scope('lstm'):
        cells_fw = [ tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(units) ]
        cells_bw = [ tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(units) ]
        (net, unused_state_f, unused_state_b) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw, cells_bw, net, dtype=tf.float32, sequence_length=None, parallel_iterations=1)
        net = tf.layers.dropout(net, 0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return net


def fc_layer(net, hiddens):
    return tf.layers.dense(net, hiddens, kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, kernel_regularizer=l2_reg2)


def flatten(labels,logits):
    with tf.variable_scope('flatten_logits'):
        pitch_labels, frame_labels = labels
        pitch_logits, frame_logits = logits

        pitch_labels = tf.cast(tf.reshape(pitch_labels, [-1, 88]), tf.float32)
        frame_labels = tf.cast(tf.reshape(frame_labels, [-1, 88]), tf.float32)

        pitch_logits = tf.cast(tf.reshape(pitch_logits, [-1, 88]), tf.float32)
        frame_logits = tf.cast(tf.reshape(frame_logits, [-1, 88]), tf.float32)
    return (pitch_labels, frame_labels), (pitch_logits, frame_logits)


# we use attack transition and whole transition to model the piano signal evolution process.
# Multi frame around onset time are labeled with one.
def onset_and_frames_loss(labels, logits):
    with tf.variable_scope('total_loss'):
        (pitch_labels, frame_labels), (pitch_logits, frame_logits) = flatten(labels, logits)
        pitch_loss = tf.nn.weighted_cross_entropy_with_logits(
            targets=tf.cast(tf.greater(pitch_labels, 0.0), tf.float32), logits=pitch_logits, pos_weight=1)
        pitch_loss = tf.reduce_sum(tf.reduce_mean(pitch_loss, axis=1))
        frame_loss = tf.nn.weighted_cross_entropy_with_logits(
            targets=frame_labels, logits=frame_logits, pos_weight=1)
        frame_loss = tf.reduce_sum(tf.reduce_mean(frame_loss, axis=1))
        
        total_loss = tf.reduce_sum([pitch_loss,  frame_loss,
            tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))], name="loss")

    return total_loss


# the metric for the onset and frame part.
def onset_and_frame_metric(labels, logits):
    (pitch_labels, frame_labels), (pitch_logits, frame_logits) = flatten(labels, logits)
    with tf.variable_scope('cast_type'):
        pitch_labels = tf.cast(tf.greater(pitch_labels, 0.0), dtype=tf.uint8)
        frame_labels = tf.cast(frame_labels, dtype=tf.uint8)
        pitch_predictions = tf.cast(tf.greater(tf.nn.sigmoid(pitch_logits),0.5), tf.uint8)
        frame_predictions = tf.cast(tf.greater(tf.nn.sigmoid(frame_logits),0.5), tf.uint8)

    def cal_f1(p, r, name):
        with tf.variable_scope(name):
            f = tf.divide(2*p[0]*r[0], p[0]+r[0]+1e-8, name='value'), tf.divide(2*p[1]*r[1], p[1]+r[1]+1e-8, name='update_op')
        return f

    pitch_p = tf.metrics.precision(labels=pitch_labels, predictions=pitch_predictions, name='pitch_p')
    pitch_r = tf.metrics.recall(labels=pitch_labels, predictions=pitch_predictions, name='pitch_r')
    pitch_f = cal_f1(pitch_p, pitch_r, 'pitch_f')

    frame_p = tf.metrics.precision(labels=frame_labels, predictions=frame_predictions, name='frame_p')
    frame_r = tf.metrics.recall(labels=frame_labels, predictions=frame_predictions, name='frame_r')
    frame_f = cal_f1(frame_p, frame_r, 'frame_f')


    tf.summary.scalar('pitch_p', pitch_p[1])
    tf.summary.scalar('pitch_r', pitch_r[1])
    tf.summary.scalar('pitch_f', pitch_f[1])

    tf.summary.scalar('frame_p', frame_p[1])
    tf.summary.scalar('frame_r', frame_r[1])
    tf.summary.scalar('frame_f', frame_f[1])

    metrics = {
        'pitch_f':pitch_f,
        'pitch_p': pitch_p,
        'pitch_r': pitch_r,
        'frame_f': frame_f,
        'frame_p': frame_p,
        'frame_r': frame_r
    }
    return metrics


# this part is use to detect onset-event and can reduce extra note errors.
# for this part is just a binary classification task, it is not mentioned
# in our paper.
def onset_model(inputs, mode): # -1, W, window_size, channels

    l2_reg1 = tf.contrib.layers.l2_regularizer(0.0)
    l2_reg2 = tf.contrib.layers.l2_regularizer(1e-4)
    kernel_initializer = tf.initializers.variance_scaling()
    bias_initializer = tf.initializers.constant(value=0.1)

    inputs = normalize(inputs)
    inputs = tf.transpose(inputs, [0, 2, 1, 3])

    with tf.variable_scope('conv_block'):

        net = tf.layers.conv2d(inputs, 10, [15, 3], padding='valid', kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(net, momentum=0.9, epsilon=1e-7, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 1], padding='same')

        net = tf.layers.conv2d(net, 20, [13, 3], padding='valid', kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(net, momentum=0.9, epsilon=1e-7, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 1], padding='same')


    with tf.variable_scope('fc'):
        net = tf.transpose(net, [0, 2, 1, 3])
        net = tf.reshape(net, [-1, 7, 80*20])
        net = tf.layers.dense(net, 256, activation=tf.nn.relu, kernel_initializer=kernel_initializer, 
          bias_initializer=bias_initializer, kernel_regularizer=l2_reg2)

    with tf.variable_scope('lstm'):
        cells_fw = [ tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(128) ]
        cells_bw = [ tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(128) ]
        (net, unused_state_f, unused_state_b) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw, cells_bw, net, dtype=tf.float32, sequence_length=None, parallel_iterations=1)

    logits = tf.layers.dense(net, 1, kernel_initializer=kernel_initializer, 
             bias_initializer=bias_initializer, kernel_regularizer=l2_reg2)

    return logits


# onset model's loss function
def onset_loss(labels, logits):

    with tf.variable_scope('weighted_loss'):
        with tf.variable_scope('cast_type'):
            labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
            logits = tf.reshape(tf.cast(logits, tf.float32), [-1, 1])

        sigmoid_loss = tf.nn.weighted_cross_entropy_with_logits(targets=tf.cast(tf.greater(labels, 1.0), tf.float32), 
                        logits=logits, pos_weight=1)
        loss = tf.add(tf.reduce_sum(tf.reduce_mean(sigmoid_loss, axis=0)),
                      tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name='loss')
    return loss


# onset model metrics
def onset_metric(labels, probs):

    with tf.variable_scope('cast_type'):
        labels = tf.reshape(tf.cast(labels,tf.float32), [-1, 1])
        probs = tf.reshape(probs, [-1, 1])
        labels = tf.cast(tf.greater(labels, 1.0), dtype=tf.uint8)
        predictions = tf.cast(tf.greater(probs, 0.5), dtype=tf.uint8)

    p = tf.metrics.precision(labels=labels, predictions=predictions)
    r = tf.metrics.recall(labels=labels, predictions=predictions)
    auc = tf.metrics.auc(labels=labels, predictions=probs)
    with tf.variable_scope('f1_score'):
        f = tf.divide(2*p[0]*r[0], p[0]+r[0]+1e-8, name='value'),tf.divide(2*p[1]*r[1], p[1]+r[1]+1e-8, name='update_op')

    tf.summary.scalar('train_p', p[1])
    tf.summary.scalar('train_r', r[1])
    tf.summary.scalar('train_f', f[1])
    tf.summary.scalar('train_auc', auc[1])

    metrics = {
        'test_f':f,
        'test_p': p,
        'test_r': r,
        'test_auc': auc,
    }
    return metrics