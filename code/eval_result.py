'''
This script use universally acknowledged metrices to evaluate the transcription results.
Mir_eval is used to accomplish this job. 
'''
import argparse
import os
import mir_eval
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--label_path', type=str, required=True)
parser.add_argument('--result_path', type=str, required=True)
parser.add_argument('--onset_tolerance', type=float, default=0.05)

def eval_result(args):

    label_path = args.label_path
    result_path = args.result_path
    onset_tolerance = args.onset_tolerance
    eval_files = [filename for filename in os.listdir(result_path)]
    filenames = []
    results = []

    for filename in eval_files:
        est_file = os.path.join(result_path, filename)
        ref_file = os.path.join(label_path, filename.split('.')[0]+'.txt')
        ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals(ref_file)
        est_intervals, est_pitches = mir_eval.io.load_valued_intervals(est_file)

        precision, recall, f1, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            mir_eval.util.midi_to_hz(ref_pitches),
            est_intervals,
            mir_eval.util.midi_to_hz(est_pitches),
            onset_tolerance=onset_tolerance)

        filenames.append(filename)
        results.append([f1, precision, recall])

    mean_result = np.mean(results, axis=0)

    zipped = zip(filenames, results)
    sorted_zipped = sorted(zipped, key=lambda x:x[0])
    unzipped = zip(*sorted_zipped)
    filenames, results = [list(e) for e in unzipped]

    for filename, (f1, p, r) in zip(filenames, results):
        print('{}\nf1: {:.4f} precision: {:.4f} recall: {:.4f}\n'
            .format(filename, f1, p, r))

    print('mean_result\nf1: {:.4f} precision: {:.4f} recall: {:.4f}\n'
        .format(mean_result[0], mean_result[1], mean_result[2]))


if __name__ == '__main__':
    args = parser.parse_args()
    eval_result(args)
