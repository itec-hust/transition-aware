from easydict import EasyDict

# 101 frame length is used as our optimal length to learn on MAESTRO dataset.
# CQT transform parameters
params = EasyDict(
    {'sr': 44100, 'hop_len': 512, 'fmin': 27.5, 'bins_per_octave': 48, 'n_bins': 356, 'mono': False, 'win_len': 9,
     'max_len': 101,
     'onset_label_len': 7})

# train params and tfrecord files config
# you need to configurate the tfrd_path as the root path to
# save onset model and pitch model training and evaluation tfrecords. 
train_params = EasyDict({'train_onset_examples': 15177059, 'train_pitch_examples': 369800, 'initial_lr': 0.0005,
                         'save_checkpoints_steps': 2000, 'decay_steps': 1000, 'decay_rate': 0.98,
                         'tfrd_path': 'need to complete!', 'parallel_num': 8})
