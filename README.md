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

Feature Extraction
Audio features can be classified into 3 categories: high-level, mid-level, and low-level
- High-level = features related to music lyrics (chords, rhythm, melody)
- Mid-level = features related to beat level attributes, pitch-like fluctuation patterns, and MFCCS


Methodology
-----------
The top 4 approaches to music genre classification are:
    1. Multiclass SVM
    2. KNNs
    3. K-Means Clustering
    4. CNNs
I guess? I'll slowly move along this list.
