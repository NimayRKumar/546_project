import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import shutil

from PIL import Image

#replace with your own if using absolute paths, or empty string if not
root = '/Users/nimay/Desktop/repos/in_the_jungle'
FIG_SIZE = (15,10)

def rename_data(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                new = file.split('_')[1]
                shutil.move(os.path.join(rootdir, subdir, file), os.path.join(rootdir, subdir, new))


def create_spectrogram(file):
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

    print("STFT hop length duration is: {}s".format(hop_length_duration))
    print("STFT window duration is: {}s".format(n_fft_duration))

    # perform stft
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # calculate abs values on complex numbers to get magnitude
    spectrogram = np.abs(stft)

    # display spectrogram
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.title("Spectrogram")

    # apply logarithm to cast amplitude to Decibels
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    fig = plt.Figure()
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)



    exit(1)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")

    # MFCCs
    # extract 13 MFCCs
    MFCCs = librosa.feature.mfcc(y = signal, sr = sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    # librosa.feature.mfcc(y=y, sr=sr)
    # MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate)

    # display MFCCs
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("MFCCs")

    # show plots
    plt.show()

file = root + '/data/cat/wav/1.wav'
#rename_data(root)
create_spectrogram(file)