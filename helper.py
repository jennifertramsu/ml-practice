import librosa

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
    X_audio : list
        Cleaned X_train list of loaded audio files.

    y_train : list
        Cleaned list of labels.
    '''
    X_audio = []

    for i in range(len(X_train)):
        path = X_train.iloc[i]
        audio, err = preprocess(path)

        if err:
            del y_train[i]
            i -= 1
        else:
            X_audio.append(audio)

    return X_audio, y_train
