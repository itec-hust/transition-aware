from pandas import read_csv
from functools import partial
from librosa import cqt, load
from glob import glob
import random
import numpy as np
import tensorflow as tf

random.seed(1)

from configs import params

# for the tfrecord files, int features
def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.ravel()))

# for the tfrecord files, float features
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.ravel()))

data_csv_path = '/home/data/maestro-v1.0.0/maestro-v1.0.0.csv'
data_root = '/home/data/maestro-v1.0.0/'

def get_annotations(mode):
    '''
    input:
      mode: the learning phase, like [train, evaluate, predict]
    output:
      annotaion: the appropriate annotations.
    '''
    csv_data = read_csv(data_csv_path)

    if mode == tf.estimator.ModeKeys.TRAIN:
        data_info = csv_data.loc[csv_data['split'] == 'train']
    elif mode == tf.estimator.ModeKeys.EVAL:
        data_info = csv_data.loc[csv_data['split'] == 'validation']
    elif mode == tf.estimator.ModeKeys.PREDICT:
        data_info = csv_data.loc[csv_data['split'] == 'test']

    wav_name = list(data_info['audio_filename'])
    wav_paths = list(map(partial(os.path.join, data_root), wav_name))

    if mode == tf.estimator.ModeKeys.PREDICT:
      return wav_paths

    label_paths = list(map(lambda x: x.replace('.wav', '.txt'), wav_paths))

    annotation=list(zip(wav_paths,label_paths))
    random.shuffle(annotation)

    return annotation


def cqt_dual(wav_path):
    '''
    input:
      wav_path: audio path.
    outuput:
      specs: CQT spectrogram.
    '''
    y, _ = load(wav_path, params.sr, params.mono)
    inner_cqt = partial(cqt, sr = params.sr, hop_length = params.hop_len, fmin = params.fmin, n_bins = params.n_bins, 
                        bins_per_octave = params.bins_per_octave)
    specs = inner_cqt(y[0]), inner_cqt(y[1])
    specs = np.abs(np.stack(specs, axis=-1)) # H, W, C

    return specs


def input_fn(data_dir, mode):
    '''
    input:
      data_dir: the input tfrecords dirpath
      mode: the learning phase in [train, evalute, predict]
    output:
      the dataset pipeline in tensorflow.
    '''

    def parser(serialized_example):

        C = 1 if params.mono else 2
        H, W= params.n_bins, params.max_len+8
        if FLAGS.network == 'onset':
            label_len = 1
            label_frame = 7
        elif FLAGS.network =='pitch':
            label_len = 88
            label_frame = params.max_len

        features = tf.parse_single_example(
            serialized_example,
            features={
                'spec': tf.FixedLenFeature([H*W*C], tf.float32),
                'pitch_label': tf.FixedLenFeature([label_frame*label_len], tf.int64),
                'frame_label':tf.FixedLenFeature([label_frame*label_len], tf.int64)
            })

        spec = tf.cast(tf.reshape(features['spec'], [H, W, C]), tf.float32)
        pitch_label = tf.cast(tf.reshape(features['pitch_label'], [label_frame, label_len]), tf.uint8)
        frame_label = tf.cast(tf.reshape(features['frame_label'], [label_frame, label_len]), tf.uint8)

        return spec, (pitch_label, frame_label)

    tfrecords_file = glob(os.path.join(data_dir, '*.tfrecords'))

    if mode == tf.estimator.ModeKeys.TRAIN:
        random.shuffle(tfrecords_file)

    with tf.variable_scope('input_pipe'):
        dataset = tf.data.TFRecordDataset(tfrecords_file,num_parallel_reads=None)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.apply( tf.contrib.data.shuffle_and_repeat(buffer_size=2048, count=FLAGS.epochs))
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parser, batch_size=FLAGS.batch_size, num_parallel_calls=None))
        dataset=dataset.prefetch(6)

    return dataset


onset_thrd = 0.5
pitch_thrd = 0.6
adjacent_time = 0.05

def note_search(probs, filename):
    '''
    input:
      probs: the concated [onset, pitch] probs
      filename: the filename for the ouput result file
    ouput:
      None. The results will be write to a text file endswith .res
    '''
    def reject_note(buffer_notes):
        notes = []
        last_onset_time = [-1] * 88
        for onset_time, midi in buffer_notes:
            if onset_time - last_onset_time[midi - 21] > adjacent_time:
                notes.append([onset_time, midi])
                last_onset_time[midi - 21] = onset_time
        return notes

    onset_prob, pitch_prob = np.split(probs, [1,], axis=-1)

    onset_mask = onset_prob > onset_thrd
    pitch_prob = pitch_prob * onset_mask

    valid_indexs = np.argwhere(onset_mask)[:, 0]

    start = valid_indexs[0]
    last = start
    buffer_notes = []

    for idx in valid_indexs[1:]:
        if idx - last > 1:
            onset_max_idx = np.argmax(onset_prob[start:last+1, 0], axis=0) + start
            pitch_max_idxs = np.argmax(pitch_prob[start:last+1, :], axis=0) + start

            for i in range(88):
                pitch_max_idx = pitch_max_idxs[i]
                if pitch_prob[pitch_max_idx, i] > pitch_thrd:
                    buffer_notes.append([(pitch_max_idx+onset_max_idx)/2*params.hop_len/params.sr, i+21])

            start=idx
        last=idx

    buffer_notes.sort(key=lambda x: x[0])
    notes = reject_note(buffer_notes)

    res_path = os.path.join(FLAGS.save_res_dir, filename.replace('.tfrecords', '.res'))
    with open(res_path,'w') as f:
        for onset, pitch in notes:
            f.write('{:.6f}\t{:.6f}\t{}\n'.format(onset,onset+0.1, pitch))