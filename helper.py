import librosa
import numpy as np
import pandas as pd

def preprocess(file):
    ''' This function takes in the path to an audio file and returns the floating-point time series.
    Parameters
    ----------
    file : str
        The path to the audio file to be loaded.

    Returns
    -------
    audio : NumPy Array
        Floating-point time series.

    err : int {0, 1}
        Error code. 
    '''
    err = 0

    try:
        audio = librosa.load(file)[0]
    except:
        print(file + " is corrupted!")
        err = 1
        audio = None
    finally:
        return audio, err

def process_train(X_train, y_train):
    ''' This function will process a list of audio files.

    Parameters
    ----------
    X_train : list
        List of path names to the training audio files.

    y_train : list
        List of labels corresponding to X_train.

    Returns
    -------
    X_audio : NumPy Array
        Cleaned X_train list of loaded audio files.

    y_audio : list
        Cleaned list of labels.
    '''
    X_audio = []
    y_audio = []

    for i in range(len(X_train)):
        path = X_train[i]
        label = y_train[i]

        audio, err = preprocess(path)

        if not err:
            X_audio.append(audio)
            y_audio.append(label)

    X_audio = np.array(X_audio, dtype='object')

    return X_audio, y_audio

def amplitude_envelope(signal, frame_size, hop_length):
    ''' This function returns the amplitude envelope of a signal.
    Parameters
    ----------
    signal : NumPy Array

    frame_size : int

    hop_length : int

    Returns
    -------
    '''

    return np.array([max(signal[i:i+frame_size]) for i in range(0, signal.size, hop_length)])

def extract(audio, FRAME_SIZE = 1024, HOP_LENGTH = 512):
    ''' This function returns a dictionary of the features for an audio file.
    Parameters
    ----------
    audio : NumPy Array

    FRAME_SIZE : int (default 1024)

    HOP_LENGTH : int (default 512)

    Returns
    -------
    features : dict

    '''

    features = {
        'Zero Crossing Rate': np.average(librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]),
        'Amplitude Envelope': np.average(amplitude_envelope(audio, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH)),
        'RMS': np.average(librosa.feature.rms(audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]),
        'Spectral Centroid': np.average(librosa.feature.spectral_centroid(audio))
    }

    return features

def feature_extract(X_audio_train):
    ''' This function extracts the features for a list of audio files.
    Parameters
    ----------
    X_audio_train : list

    Returns
    -------
    X_features_train : Pandas DataFrame
    '''

    rows = []

    for row in X_audio_train:

        features = extract(row)
        rows.append(features)

    X_features_train = pd.DataFrame(rows)

    return X_features_train