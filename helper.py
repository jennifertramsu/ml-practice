import librosa

def preprocess(file):
    ''' 
    Parameters
    ----------
    file : str
        The path to the audio file to be loaded.

    Returns
    -------
    audio :

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


    Returns
    -------
    
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
