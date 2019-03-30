from sklearn import neighbors
import numpy as np
from wav_to_mfcc import wav_to_mfcc
from voice_recorder_scr import record_to_file
from time import sleep
import pickle
import speech_recognition as sr
import textClassifier

with open('voiceClassifiers/Neural Net.pkl', 'rb') as f:
    gaussianProcess = pickle.load(f)

file_name = 'recording.wav'

print('Say something')

record_to_file(file_name)

print('Done! Processing...')

voice_feature = wav_to_mfcc(file_name)
voice_prediction = {'NOT DEPRESSED': 'HAPPY', 'DEPRESSED': 'DEPRESSED'}[gaussianProcess.predict([voice_feature])[0]]

r = sr.Recognizer()
with sr.AudioFile(file_name) as source:
    audio = r.record(source)
text = r.recognize_google(audio)
print('What you said:', text)
print('\n')
text_prediction = textClassifier.textPredict(text)

print('You sound {} and you are speaking about {} things.'.format(voice_prediction, text_prediction))
