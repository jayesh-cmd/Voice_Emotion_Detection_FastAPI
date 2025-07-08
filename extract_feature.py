import librosa
import numpy as np

def extraction(filepath , max_len = 174 , n_mfcc = 40):
    audio , sr = librosa.load(filepath , sr = 22050 , res_type="kaiser_fast")
    mfcc = librosa.feature.mfcc(y = audio , sr = sr , n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc , ((0,0) , (0 , pad)) , mode = 'constant')
    else:
        mfcc = mfcc[: , :max_len]

    return mfcc.T