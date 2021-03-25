'''
This script is use to predict pitch probs in the input spectrogram 
segments. The output probs shape is [time_steps, 88] and will be handled
with the ouput probs of the onset model to get the final (onset, pitch) results.
'''
import os
import tensorflow as tf
import numpy as np
import argparse
from multiprocessing import Pool
from functools import partial
from configs import params
from tools import cqt_dual
from tools import _float_feature
from models import normalize, acoustic_model, lstm_layer, fc_layer

parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', type=str, required=True)
parser.add_argument('--predict_dir', type=str, required=True, 
        help='the pathdir to store tfrecord files')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--save_raw_dir', type=str, required=True, 
        help='the pathdir to store raw pitch probs in a text format')
parser.add_argument('--ckpt_path', type=str, required=True, 
        help='the checkpoint file path restore to predict pitch probs')

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def generate_test_tfrd(wav_path, predict_dir):
    spec = cqt_dual(wav_path)
    max_len=params.max_len

    data_len = int(np.ceil(spec.shape[1]/max_len)*max_len)
    end_pad = data_len-spec.shape[1]
    data_num = int(data_len/max_len)
    spec = np.pad(spec, ((0,0),(0, end_pad), (0,0))) #[n_bins, len, 2]

    pad_len=4
    spec = np.pad(spec, ((0, 0), (pad_len, pad_len), (0,0)))

    to_example = lambda spec: tf.train.Example(features=tf.train.Features(feature={'spec': _float_feature(spec)}))
    spec_offset = int((max_len+8)/2)

    suffix='.'+os.path.basename(wav_path).split('.')[1]

    with tf.python_io.TFRecordWriter(os.path.join(predict_dir, os.path.basename(wav_path).replace(suffix, '.tfrecords'))) as w:
        for i in range(data_num):
            j = spec_offset+i*max_len
            example = to_example(spec[:, j-spec_offset:j+spec_offset+1]).SerializeToString()
            w.write(example)

def wav_to_tfrd(wav_paths, predict_dir):
    p = Pool(16)
    generate_test_tfrd = partial(generate_test_tfrd, predict_dir=predict_dir)
    result=p.map_async(generate_test_tfrd, wav_paths)
    result.get()
    p.close()
    p.join()

def input_fn(data_path, batch_size):
    def parser(serialized_example):
        C = 1 if params.mono else 2
        H, W = params.n_bins, params.max_len+8
        features = tf.parse_single_example(
            serialized_example,
            features={
                'spec': tf.FixedLenFeature([H*W*C], tf.float32),
            })
        spec = tf.cast(tf.reshape(features['spec'], [H, W, C]), tf.float32)
        return spec

    with tf.variable_scope('input_pipe'):
        dataset = tf.data.TFRecordDataset(data_path, num_parallel_reads=None)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parser, batch_size=batch_size, num_parallel_calls=None))
        dataset=dataset.prefetch(6)

    return dataset

def model_fn(features, labels, mode, params):

    with tf.variable_scope('pitch'):
        inputs = normalize(features)
        with tf.variable_scope('pitch'):
            net1 = acoustic_model(inputs, mode) # [batch_size, time_len, 1024] 
            net1= lstm_layer(net1, 512, mode) # [batch_size, time_len, 1024] 
            pitch_logits = fc_layer(net1, 88)
     
    old_pitch_scope = ['pitch/']
    new_pitch_scope = ['pitch/' + x for x in old_pitch_scope]
    pitch_scope = {old_scope: new_scope for old_scope, new_scope in zip(old_pitch_scope, new_pitch_scope)}

    predictions = {
        'pitch_probs': tf.nn.sigmoid(pitch_logits, name='sigmoid_pitch')  # [Batch_size, len, 88]
    }

    model_paths = params['ckpt_path']
    tf.train.init_from_checkpoint(model_paths, pitch_scope)

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    
def main(args):

    audio_files = os.listdir(args.audio_dir)
    suffix = '.'+audio_files[0].split('.')[1]
    
    audio_files = [filename for filename in audio_files if filename.replace(suffix,'.pitch_raw') not in os.listdir(args.save_raw_dir)]
    tfrd_files = [filename.replace(suffix,'.tfrecords') for filename in audio_files]

    audio_files_needed = [filename for filename in audio_files if filename.replace(suffix,'.tfrecords') not in os.listdir(args.predict_dir)]
    wav_files = [os.path.join(args.audio_dir, filename) for filename in audio_files_needed]
    if wav_files:
        wav_to_tfrd(wav_files, args.predict_dir)

    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    run_config = tf.estimator.RunConfig(session_config=sess_config, save_summary_steps=100, log_step_count_steps=500)

    model = tf.estimator.Estimator(model_fn=model_fn, params={'ckpt_path': args.ckpt_path}, config=run_config, model_dir=None)

    for tfrd_file in tfrd_files:

        print(tfrd_file)

        tfrd_path = os.path.join(args.predict_dir, tfrd_file)
        predictions = model.predict(input_fn=lambda: input_fn(tfrd_path, args.batch_size))

        pitch_probs_array = []

        for idx, prediction in enumerate(predictions):

            pitch_probs = prediction['pitch_probs']
            pitch_probs_array.append(pitch_probs)

        pitch_probs_array = np.concatenate(pitch_probs_array, axis=0)
            
        probs = pitch_probs_array
        raw_path = os.path.join(args.save_raw_dir, tfrd_file.replace('.tfrecords', '.pitch_raw'))
        np.savetxt(raw_path, probs, fmt='%.6f', delimiter='\t')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)