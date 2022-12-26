# Music Classification

Project Statement
-----------------
What makes a music genre unique?
Originally, I wanted to explore different music periods in history (Baroque, Renaissance, Classical, etc.) but I can't seem to find any existing datasets for this and I can't be bothered to make my own.

This project will explore how audio is processed and how to represent music in a way that is understood by computers to make predictions.

Dataset
-------
The GTZAN genre collection dataset contains 1000 audio files belonging to 10 different classes.
Each audio file is in .wav format (extension).
The classes are:

    1. Blues
    2. Hip-hop
    3. Classical
    4. Pop
    5. Disco
    6. Country
    7. Metal
    8. Jazz
    9. Reggae
    10. Rock

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Data Preprocessing
------------------
How do we represent and characterize music?

<h3><a href=https://www.analyticsvidhya.com/blog/2022/03/music-genre-classification-project-using-machine-learning-techniques/>
Feature Extraction
</a></h3>

Audio features can be classified into 3 categories: high-level, mid-level, and low-level
- High-level = features related to music lyrics (chords, rhythm, melody)
- Mid-level = features related to beat level attributes, pitch-like fluctuation patterns, and MFCCS
- Low-level = features including amplitude envelope, energy, zero-crossing rate, etc.

<h3><a ref=https://towardsdatascience.com/learning-from-audio-spectrograms-37df29dba98c>
Audio Representation
</a></h3>

- Spectrogram = photographic or other visual or electronic representation of a spectrum
    - Represents time, frequency, and amplitude in one graph
        - Frequency vs time, amplitude is brightness/colour
    - How to create?
        1. Split the audio into overlapping chunks or windows
        2. Perform the short time fourier transformation on each window (absolute value)
        3. Each resulting window has a vertical line presenting the magnitude vs. frequency
        4. Take the resulting window and convert to decibels
        5. Lay these windows back into the length of the original song and display the output

https://towardsdatascience.com/learning-from-audio-the-mel-scale-mel-spectrograms-and-mel-frequency-cepstral-coefficients-f5752b6324a8

Mel Scale

- Mel scale = logarithmic transformation of a signal's frequency
    - Sounds of equal distance on the Mel scale are perceived to be of equal distance to humans
    $ m = 1127 \bullet ln(1 + \frac({f}{700}))$

Mel Spectrograms

- Spectrograms that visualize sounds on the Mel scale as opposed to the frequency domain

Mel Frequency Cepstral Coefficients

    1. Convert from Hertz to Mel scale
    2. Take logarithm of Mel representation of audio
    3. Take logarithmic magnitude and use Discrete Cosine Transformation
    4. This result creates a spectrum over Mel frequencies as opposed to time, thus creating MFCCs
    
- In ML, the number of MFCCs used is a hyperparameter

https://medium.com/@tanveer9812/mfccs-made-easy-7ef383006040

- MFCCs are a compact representation of the spectrum of an audio signal

Methodology
-----------
The top 4 approaches to music genre classification are:

    1. Multiclass SVM
    2. KNNs
    3. K-Means Clustering
    4. CNNs
    
I guess? I'll slowly move along this list.

<h2> SVM </h2>

<h3>First Trial</h3>
Still just exploring how to extract features from audio files. For now, the following time-domain features are used:

    1. Mean zero-crossing rate
    2. Mean amplitude envelope
    3. Mean RMS
    
Accuracy: 29% LOL
