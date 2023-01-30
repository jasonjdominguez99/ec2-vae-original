# preprocess_midi_data.py
#
# main source code for preprocess the Nottingham dataset
# melody and chord midi data files, in accordance with the
# 'DEEP MUSIC ANALOGY VIA LATENT REPRESENTATION
# DISENTANGLEMENT' paper, for use with the ec-squared vae model
    
# Code from CMT
    
import glob
import os
import random
import argparse
import pickle
import numpy as np
import pretty_midi as pm
from tqdm import tqdm
from scipy.sparse import csc_matrix


def pad_pianorolls(pianoroll, timelen):
    if pianoroll.shape[1] < timelen:
        pianoroll = np.pad(pianoroll, ((0, 0), (0, timelen - pianoroll.shape[1])),
                           mode="constant", constant_values=0)
    return pianoroll


def preprocess(root_dir, midi_dir, save_file_path, num_bars, frame_per_bar, pitch_range=48, shift=False,
                            beat_per_bar=4, bpm=120, data_ratio=(0.8, 0.1, 0.1)):

    instance_len = frame_per_bar * num_bars
    # print(f"Instance length: {instance_len}")
    # stride = int(instance_len / 2)
    stride = instance_len
    # print(f"Stride: {stride}")
    # Default : frame_per_second=8, unit_time=0.125
    frame_per_second = (frame_per_bar / beat_per_bar) * (bpm / 60)
    unit_time = 1 / frame_per_second

    song_list = sorted(glob.glob(os.path.join(root_dir, midi_dir, '*')))
    # print(f"Song list: {song_list}")
    # print(f"Song list length: {len(song_list)}")
    # print(f"midi file paths: {os.path.join(root_dir, midi_dir, '*.mid')}")
    midi_files = sorted(glob.glob(os.path.join(root_dir, midi_dir, '*.mid')))
    print(f"Number of midi files: {len(midi_files)}")

    num_eval = int(len(song_list) * data_ratio[1])
    print(f"num_eval: {num_eval}")
    num_test = int(len(song_list) * data_ratio[2])
    print(f"num test: {num_test}")
    random.seed(0)
    eval_test_cand = set([song.split('/')[-1] for song in song_list])
    eval_set = random.choices(list(eval_test_cand), k=num_eval)
    test_set = random.choices(list(eval_test_cand - set(eval_set)), k=num_test)

    pitches_train = []
    chords_train = []
    pitches_eval = []
    chords_eval = []
    pitches_test = []
    chords_test = []

    for midi_file in tqdm(midi_files, desc="Processing"):
        filename = midi_file.split('/')[-1]

        if filename in eval_set:
            mode = "eval"
        elif filename in test_set:
            mode = "test"
        else:
            mode = "train"

        if shift:
            pitch_shift = range(-5, 7)
        else:
            pitch_shift = [0]
            
        for k in pitch_shift:
            melody_only = False
            midi = pm.PrettyMIDI(midi_file)
            if len(midi.instruments) < 1:
                # print("No instruments")
                continue
            elif len(midi.instruments) < 2:
                # print("Only one instrument (assume melody)")
                melody_only = True
            
            on_midi = pm.PrettyMIDI(midi_file)
            off_midi = pm.PrettyMIDI(midi_file)
            note_instrument = midi.instruments[0]
            onset_instrument = on_midi.instruments[0]
            offset_instrument = off_midi.instruments[0]
            
            for note, onset_note, offset_note in zip(note_instrument.notes, onset_instrument.notes, offset_instrument.notes):
                if k != 0:
                    note.pitch += k
                    onset_note.pitch += k
                    offset_note.pitch += k
                note_length = offset_note.end - offset_note.start
                onset_note.end = onset_note.start + min(note_length, unit_time)
                offset_note.end += unit_time
                offset_note.start = offset_note.end - min(note_length, unit_time)
            
            # original code, which resulted in missing last note
            # pianoroll = note_instrument.get_piano_roll(fs=frame_per_second)
            pianoroll = note_instrument.get_piano_roll(
                times=np.arange(0, midi.get_end_time(), 1./frame_per_second)
            )
            # print(f"pianoroll shape: {pianoroll.shape}")
            onset_roll = onset_instrument.get_piano_roll(
                times=np.arange(0, midi.get_end_time(), 1./frame_per_second)
            )
            # print(f"onset_roll shape: {onset_roll.shape}")
            offset_roll = offset_instrument.get_piano_roll(
                times=np.arange(0, midi.get_end_time(), 1./frame_per_second)
            )
            # print(f"offset_roll shape: {offset_roll.shape}")

            timelen = min(pianoroll.shape[1], offset_roll.shape[1])

            if not melody_only:
                chord_instrument = midi.instruments[1]
                for chord_note in chord_instrument.notes:
                    if k != 0:
                        chord_note.pitch += k
                    chord_note.end = chord_note.start + unit_time
                chord_onset = chord_instrument.get_piano_roll(fs=frame_per_second)
                
                chord_onset = pad_pianorolls(chord_onset, timelen)
                chord_onset[chord_onset > 0] = 1

            pianoroll = pad_pianorolls(pianoroll, timelen)
            onset_roll = pad_pianorolls(onset_roll, timelen)
            offset_roll = pad_pianorolls(offset_roll, timelen)

            pianoroll[pianoroll > 0] = 1
            onset_roll[onset_roll > 0] = 1
            offset_roll[offset_roll > 0] = 1

            # print(f"timelen: {timelen}")
            # print(f"instance_len {instance_len}")
            # for i in range(0, timelen - (instance_len + 1), stride):
            for i in range(1):
                pitch_list = []
                chord_list = []
                
                if not melody_only:
                    chord_inst = chord_onset[:, i:(i + instance_len + 1)]

                pianoroll_inst = pianoroll[:, i:(i+instance_len+1)]
                onset_inst = onset_roll[:, i:(i+instance_len+1)]

                if not melody_only:
                    if len(chord_inst.nonzero()[1]) < 4:
                        continue

                rhythm_idx = np.minimum(np.sum(pianoroll_inst.T, axis=1), 1) + np.minimum(np.sum(onset_inst.T, axis=1), 1)
                rhythm_idx = rhythm_idx.astype(int)
                # print(f"rhythm_idx: {rhythm_idx}")
                # print(rhythm_idx.nonzero()[0].size)
                # print((instance_len // 4))
                # If more than 75% is not-playing, do not make instance
                if rhythm_idx.nonzero()[0].size < (instance_len // 4):
                    continue
                
                if pitch_range == 128:
                    base_note = 0
                else:
                    highest_note = max(onset_inst.T.nonzero()[1])
                    lowest_note = min(onset_inst.T.nonzero()[1])
                    base_note = 12 * (lowest_note // 12)
                    if highest_note - base_note >= pitch_range:
                        continue

                if not melody_only:
                    prev_chord = np.zeros(12)
                cont_rest = 0
                prev_onset = 0
                
                for t in range(instance_len):
                    if t in onset_inst.T.nonzero()[0]:
                        # note is an onset
                        pitch_list.append(onset_inst[:, t].T.nonzero()[0][0] - base_note)
                        if (t != onset_inst.T.nonzero()[0][0]) and abs(onset_inst[:, t].T.nonzero()[0][0] - base_note - prev_onset) > 12:
                            cont_rest = 30
                            break
                        else:
                            prev_onset = onset_inst[:, t].T.nonzero()[0][0] - base_note
                            cont_rest = 0
                    elif rhythm_idx[t] == 1:
                        # note is a held note
                        pitch_list.append(pitch_range)
                        # pitch_list.append(pitch_list[-1])
                    elif rhythm_idx[t] == 0:
                        # note is a rest
                        pitch_list.append(pitch_range + 1)
                        cont_rest += 1
                        if cont_rest >= 30:
                            break
                    else:
                        print(filename, i, t, rhythm_idx[t], onset_inst.T.nonzero())

                    if not melody_only:
                        if len(chord_inst[:, t].nonzero()[0]) != 0:
                            prev_chord = np.zeros(12)
                            for note in sorted(chord_inst[:, t].nonzero()[0][1:] % 12):
                                prev_chord[note] = 1
                        chord_list.append(prev_chord)
                    else:
                        chord_list.append(np.zeros(12))

                if (cont_rest >= 30) or (len(set(pitch_list)) <= 5):
                    continue

                # convert pitch list to one-hot vectors with additional held-note and rest info
                # size N x 130, 128 pitches, 1 held-note and 1 rest        
                pitch_info = []
                # print(f"pitch_list: {pitch_list}")
                for pitch in pitch_list:
                    one_hot_pitch = np.zeros(pitch_range + 2)
                    one_hot_pitch[pitch] = 1.
                    pitch_info.append(one_hot_pitch)
                
                pitch_info = np.array(pitch_info)
                chord_result = np.array(chord_list)
                
                if mode == "test":
                    pitches_test.append(pitch_info)
                    chords_test.append(chord_result)
                elif mode == "eval":
                    pitches_eval.append(pitch_info)
                    chords_eval.append(chord_result)
                else:
                    pitches_train.append(pitch_info)
                    chords_train.append(chord_result)
    
    data_test = {
        'pitch': np.array(pitches_test),
        'chord': np.array(chords_test)
    }
    
    data_eval = {
        'pitch': np.array(pitches_eval),
        'chord': np.array(chords_eval)
    }
    
    data_train = {
        'pitch': np.array(pitches_train),
        'chord': np.array(chords_train)
    }
    
    print()
    print("test:")
    print(f"pitches shape: {np.array(pitches_test).shape}")
    print(f"chord_result shape: {np.array(chords_test).shape}")
    print()
    print("eval:")
    print(f"pitches shape: {np.array(pitches_eval).shape}")
    print(f"chord_result shape: {np.array(chords_eval).shape}")
    print()
    print("train:")
    print(f"pitches shape: {np.array(pitches_train).shape}")
    print(f"chord_result shape: {np.array(chords_train).shape}")
    print()
    
    # save data here
    np.save(save_file_path + "-test", data_test)
    np.save(save_file_path + "-eval", data_eval)
    np.save(save_file_path + "-train", data_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./ec_squared_vae/nottingham_dataset/MIDI')
    parser.add_argument('--midi_dir', type=str, default='melody_and_chords')
    
    # parser.add_argument('--root_dir', type=str, default='./ec_squared_vae')
    # parser.add_argument('--midi_dir', type=str, default='generation/source')
    # parser.add_argument('--midi_dir', type=str, default='generation/target')
    
    parser.add_argument('--num_bars', type=int, default=2)
    parser.add_argument('--frame_per_bar', type=int, default=16)
    parser.add_argument('--pitch_range', type=int, default=128)
    parser.add_argument('--shift', dest='shift', action='store_true')

    args = parser.parse_args()
    root_dir = args.root_dir
    midi_dir = args.midi_dir
    num_bars = args.num_bars
    frame_per_bar = args.frame_per_bar
    pitch_range = args.pitch_range
    shift = args.shift
    
    date = "2023-01-29"
    save_file_path = f"ec_squared_vae/processed_data_{date}"
    # save_file_path = "ec_squared_vae/generation/test_data"
    # save_file_path = "ec_squared_vae/generation/source/2-bar-source-melody-and-chords-figure-4"
    # save_file_path = "ec_squared_vae/generation/target/2-bar-rhythm-target-2-figure-8a"

    preprocess(root_dir, midi_dir, save_file_path,
               num_bars, frame_per_bar, pitch_range,
               shift)
    
    print("Preprocessed and saved!")
    