import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import shutil

from PIL import Image

#replace with your own if using absolute paths, or empty string if not
data_dir = '/Users/nimay/Desktop/repos/in_the_jungle/data'
FIG_SIZE = (15,10)

def rename_data(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                new = file.split('_')[1]
                shutil.move(os.path.join(rootdir, subdir, file), os.path.join(rootdir, subdir, new))


def create_spectrogram(file, out_path):
    # load audio file with Librosa
    signal, sample_rate = librosa.load(file, sr=22050)

    # WAVEFORM
    # display waveform
    plt.figure(figsize=FIG_SIZE)
    librosa.display.waveshow(signal, sr=sample_rate, color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")

    # FFT -> power spectrum
    # perform Fourier transform
    fft = np.fft.fft(signal)

    # calculate abs values on complex numbers to get magnitude
    spectrum = np.abs(fft)

    # create frequency variable
    f = np.linspace(0, sample_rate, len(spectrum))

    # take half of the spectrum and frequency
    left_spectrum = spectrum[:int(len(spectrum)/2)]
    left_f = f[:int(len(spectrum)/2)]

    # plot spectrum
    plt.figure(figsize=FIG_SIZE)
    plt.plot(left_f, left_spectrum, alpha=0.4)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum")

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

    file_path = file.split("/")[-1][:-4]
    fig = plt.Figure()

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

    exit(1)

subdirs = next(os.walk(data_dir))[1]
for sd in subdirs:
    out_path = "{}/{}".format(data_dir, sd)

    for file in os.listdir("{}/wav".format(out_path)):
        create_spectrogram("{}/wav/{}".format(out_path, file), out_path)