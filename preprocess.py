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
from sklearn.model_selection import train_test_split

with open("./data_dir_path.txt") as f:
    data_dir = f.read()

label_encoding = json.load(open('./label_encoding.JSON'))
FIG_SIZE = (15,10)

def plot_signal(file):
    signal, sample_rate = librosa.load(file, sr=22050)

    plt.figure(figsize=FIG_SIZE)
    librosa.display.waveshow(signal, sr=sample_rate, color="blue")
    plt.show()

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
    plt.show()

    # Save to MFCC
    MFCCs = librosa.feature.mfcc(y = signal, sr = sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
    plt.show()


def rename_data(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                new = file.split('_')[1]
                shutil.move(os.path.join(rootdir, subdir, file), os.path.join(rootdir, subdir, new))


def create_signal(file, out_path):
    file_path = file.split("/")[-1][:-4]
    fig = plt.Figure()

    # load audio file with Librosa
    signal, sample_rate = librosa.load(file, sr=22050)
    return signal


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
            sig = create_signal("{}/wav/{}".format(out_path, file), out_path)
            sig = pad_or_trim_audio(sig, 22050 * 2)
            df.loc[len(df)] = [sig, label_encoding[sd]]

    # df.to_csv('./csv/dataset.csv')
    return df

def get_spectrogram(signal, hop_length=512, n_fft=2048):
  stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
  spectrogram = np.abs(stft)
  return librosa.amplitude_to_db(spectrogram)

def get_mfcc(signal, n_mfcc=13):
    return librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc)

if __name__=="__main__": 
    df = create_signal_dataframe()
    # aug_list = [add_gaussian_noise, add_air_absorption]
    # df = create_augmentations(df, aug_list)
    # data = df["signal"].to_numpy() # 875 x 44100
    signal = np.vstack(df["signal"])
    label = np.array(df["label"])
    x_train, x_test, y_train, y_test = train_test_split(signal, label, test_size=0.2)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    spectrogram_train = get_spectrogram(x_train)
    print(spectrogram_train.shape)

    mfcc_train = get_mfcc(x_train)
    print(mfcc_train.shape)