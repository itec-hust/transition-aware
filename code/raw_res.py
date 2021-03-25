'''
This script is use to convert the raw onset probs (shape [time_steps, 1]) and the raw pitch probs
(shape [time_step, 88]) to the final (onset, pitch) sequence result. For the convience of evalutaton,
we ouput (onset, onset+0.1 pitch) as our ouput. So the second column is not offset as expeceted, and
we don't predict offset in our model.
'''
import argparse
import os
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('-res','--result_path',type=str,required=True,
  help='Path to save predict result files.')
parser.add_argument('-raw_pitch','--raw_pitch_path',type=str,required=True,
  help='Path to store raw files.')
parser.add_argument('-raw_onset','--raw_onset_path',type=str,required=True,
  help='Path to store raw files.')

onset_thrd = 0.5
pitch_thrd = 0.6
adjacent_time = 0.05
hop_len =512
sr = 44100

def reject_note(buffer_notes):
    notes = []
    last_onset_time = [-1] * 88
    for onset_time, midi in buffer_notes:
        if onset_time - last_onset_time[midi - 21] > adjacent_time:
            notes.append([onset_time, midi])
            last_onset_time[midi - 21] = onset_time
    return notes

def note_search(data,file_path):
    onset_prob, pitch_prob = np.split(data, [1,], axis=-1)

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
                    buffer_notes.append([(pitch_max_idx+onset_max_idx)/2*hop_len/sr, i+21])

            start=idx
        last=idx

    buffer_notes.sort(key=lambda x: x[0])
    notes = reject_note(buffer_notes)

    with open(file_path,'w') as f:
        for onset, pitch in notes:
            f.write('{:.6f}\t{:.6f}\t{}\n'.format(onset,onset+0.1, pitch))


def raw_res(args):
  result_dir=args.result_path
  filenames=[os.path.join(args.raw_pitch_path,filename) for filename in os.listdir(args.raw_pitch_path) 
              if filename.replace('.pitch_raw','.res') not in os.listdir(args.result_path)]
  for filename in filenames:
    filename_basename=os.path.basename(filename)
    print('<............... predict %s ...............>\n'%filename_basename)
    pitch_data=np.loadtxt(filename)
    onset_data=np.loadtxt(os.path.join(args.raw_onset_path,filename_basename.replace('.pitch_raw','.onset_raw')))
    onset_data=np.reshape(onset_data,[-1,1])
    data_len=onset_data.shape[0]
    data=np.concatenate([onset_data,pitch_data[:data_len]],axis=-1)
    file_path=os.path.join(result_dir,filename_basename.replace('.pitch_raw','.res'))
    note_search(data,file_path)

if __name__=='__main__':

  args = parser.parse_args()
  raw_res(args)