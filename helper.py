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
        audio = librosa.load(file)
    except:
        print(file + " is corrupted!")
        err = 1
        audio = None
    finally:
        return audio, err

def process_train(X_train, y_train):
    ''' This function will process the training files.

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

    y_clean : list
        Cleaned list of labels.
    '''
    X_audio = []
    errs = []
    
    for i in range(len(X_train)):
        path = X_train[i]
        audio, err = preprocess(path)

        if err:
            errs.append(i)
        else:
            X_audio.append(audio[0])

    X_audio = np.array(X_audio, dtype='object')
    y_clean = []

    for j in range(len(y_train)):
        if j in errs:
            continue
        else:
            y_clean.append(y_train[j])

    return X_audio, y_clean

def amplitude_envelope(signal, frame_size, hop_length):
    '''
    '''

    return np.array([max(signal[i:i+frame_size]) for i in range(0, signal.size, hop_length)])

def extract(audio):
    '''
    Parameters
    ----------
    audio : NumPy Array

    Returns
    -------
    features : dict

    '''
    # Time-Domain Features

    FRAME_SIZE = 1024
    HOP_LENGTH = 512

    features = {
        'Zero Crossing Rate': np.average(librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]),
        'Amplitude Envelope': np.average(amplitude_envelope(audio, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH)),
        'RMS': np.average(librosa.feature.rms(audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0])
    }

    return features

def feature_extract(X_audio_train):
    '''
    Parameters
    ----------
    X_audio_train : 

    Returns
    -------
    X_features_train : 
    '''

    rows = []

    for row in X_audio_train:

        features = extract(row)
        rows.append(features)

    X_features_train = pd.DataFrame(rows)

    return X_features_train