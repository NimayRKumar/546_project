import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import os
import shutil
import json
from ast import literal_eval
from audiomentations import AddGaussianNoise, AirAbsorption, ApplyImpulseResponse, BandPassFilter, GainTransition, RepeatPart, TimeStretch, TanhDistortion

with open("./data_dir_path.txt") as f:
    data_dir = f.read()

label_encoding = json.load(open('./label_encoding.JSON'))
FIG_SIZE = (15,10)

def rename_data(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                new = file.split('_')[1]
                shutil.move(os.path.join(rootdir, subdir, file), os.path.join(rootdir, subdir, new))

# Pandas df: one column for signal, one for label
# TODO: save padded signal
def create_spectrogram(file, out_path):
    file_path = file.split("/")[-1][:-4]
    fig = plt.Figure()

    # load audio file with Librosa
    signal, sample_rate = librosa.load(file, sr=22050)
    return signal

    plt.figure(figsize=FIG_SIZE)
    librosa.display.waveshow(signal, sr=sample_rate, color="blue")
    plt.savefig("{}/signal/{}".format(out_path, file_path))

    # FFT -> power spectrum
    fft = np.fft.fft(signal)

    # calculate abs values on complex numbers to get magnitude
    spectrum = np.abs(fft)

    # create frequency variable
    f = np.linspace(0, sample_rate, len(spectrum))

    # STFT -> spectrogram
    hop_length = 512 # in num. of samples
    n_fft = 2048 # window in num. of samples

    # calculate duration hop length and window in seconds
    hop_length_duration = float(hop_length)/sample_rate
    n_fft_duration = float(n_fft)/sample_rate

    # perform stft
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # calculate abs values on complex numbers to get magnitude
    spectrogram = np.abs(stft)

    # Save to DB Spectrogram
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
    plt.savefig("{}/db_spectro/{}".format(out_path, file_path))

    # Save to MFCC
    MFCCs = librosa.feature.mfcc(y = signal, sr = sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
    plt.savefig("{}/mfcc/{}".format(out_path, file_path))

# Target length = 22050 * num_seconds
def pad_or_trim_audio(signal, target_length):
  # print(len(signal))
  if len(signal) < target_length:
    # print("padded")
    signal = np.pad(signal, (0, target_length - len(signal)))
  if len(signal) > target_length:
    # print("trimmed")
    signal = signal[:target_length]
  return signal


def add_gaussian_noise(signal):
    transform = AddGaussianNoise(
        min_amplitude=0.001,
        max_amplitude=0.015,
        p=1.0
    )
    augmented_sound = transform(signal, sample_rate=22050)
    return augmented_sound

def add_air_absorption(signal):
    transform = AirAbsorption(
        min_distance=10.0,
        max_distance=50.0,
        p=1.0,
    )
    augmented_sound = transform(signal, sample_rate=22050)
    return augmented_sound

# maybe optional depending on whether we have impulse sound
def add_impulse_response(signal):
    transform = ApplyImpulseResponse(ir_path="/path/to/sound_folder", p=1.0)
    augmented_sound = transform(signal, sample_rate=22050)
    return augmented_sound

def add_band_pass_filter(signal):
    transform = BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=1.0)
    augmented_sound = transform(signal, sample_rate=22050)
    return augmented_sound

def gain_transition(signal):
    transform = GainTransition()
    augmented_sound = transform(signal, sample_rate=22050)
    return augmented_sound

def repeat_part(signal):
    transform = RepeatPart(mode="replace", p=1.0)
    augmented_sound = transform(signal, sample_rate=22050)
    return augmented_sound


def time_stretch(signal):
    transform = TimeStretch(
        min_rate=0.8,
        max_rate=1.25,
        leave_length_unchanged=True,
        p=1.0
    )
    augmented_sound = transform(signal, sample_rate=22050)
    return augmented_sound

def tanh_distortion(signal):
    transform = TanhDistortion(
        min_distortion=0.01,
        max_distortion=0.7,
        p=1.0
    )
    augmented_sound = transform(signal, sample_rate=22050)
    return augmented_sound

def create_augmentations(df, aug_list):
    for i in range(df.shape[0]):
        for aug_fxn in aug_list:
            sig = aug_fxn(df.iloc[i,0])
            df.loc[len(df)] = [list(sig), df.iloc[i, 1]]
    return df

def create_signal_dataframe(): 
    df = pd.DataFrame(columns=["signal", "label"])
    subdirs = next(os.walk(data_dir))[1]
    for sd in subdirs:
        out_path = "{}/{}".format(data_dir, sd)

        for file in os.listdir("{}/wav".format(out_path)):
            sig = create_spectrogram("{}/wav/{}".format(out_path, file), out_path)
            sig = pad_or_trim_audio(sig, 22050 * 2)
            df.loc[len(df)] = [sig, label_encoding[sd]]

    # df.to_csv('./csv/dataset.csv')
    return df
  

if __name__=="__main__": 
    df = create_signal_dataframe()
    aug_list = [add_gaussian_noise, add_air_absorption]
    df = create_augmentations(df, aug_list)
    print(df.iloc[2624])

