from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf

#This file tests the HPF performed in the MFCC preprocessing stage


fc = 80 # cut off in Hertz (cancel background noise)
fs = 16000 # specify wanted sampling frequency

b, a = signal.butter(5, fc, btype='high', analog = False, fs=fs)
# order 5, cutoff = fc, high pass, fs = 16000

w, H = signal.freqz(b, a, fs=fs) 
#determine H(w), w: [0, fs/2)
#wnorm = w/(fs/2) # norm w to go from [0,1)
plt.plot(w, 20 * np.log10(abs(H))) #graph H(w)
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency Hz')
plt.ylabel('Amplitude [dB]')
plt.axis([0,500,-100,30])
#plt.show()

#testing
filePath = r"C:\Users\camer\timit\TIMIT\TRAIN\DR6\MEAL0\SA2.WAV" 
x,sr=librosa.load(filePath, sr = None)
y = signal.lfilter(b,a,x)
outputfile = r"C:\Users\camer\test.WAV" 
sf.write(outputfile,y,sr)