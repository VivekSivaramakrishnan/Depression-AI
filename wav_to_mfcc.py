import librosa
import numpy as np


def normalize(lst):
  mean = np.mean(lst)
  sd = np.var(lst)**0.5
  return [(x - mean) / sd for x in lst]


def wav_to_mfcc(file_name):
  # handle exception to check if there isn't a file which is corrupted
  try:
    # here kaiser_fast is a technique used for faster extraction
    x, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    # we extract mfcc feature from data
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)

    feature = mfccs.tolist()

  except Exception as e:
    print("Error encountered while parsing file: ", file_name)

    feature = None

  finally:

    return normalize(feature)
