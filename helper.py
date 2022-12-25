import librosa

def preprocess(files, sr):
    ''' This function will take in a list of paths for audio files and output them as floating-point time series.

    Parameters
    ----------

    Returns
    -------

    '''

    for path in files:
        audio = librosa.load(path, sr=sr)
        print(audio)