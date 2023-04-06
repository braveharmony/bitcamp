import numpy as np

import librosa.display
import matplotlib.pyplot as plt
import matplotlib as mpl

t=np.linspace(-10,10,1000,endpoint=False)

x=np.sin(2*np.pi*0*t)
for freq in range(1,10):
    x=x+np.sin(2*np.pi*freq*t)

fs = 1 / (t[1] - t[0])

import librosa
mfcc = librosa.feature.mfcc(y=x, sr=int(fs), n_mfcc=30
                            , n_fft=100, hop_length=50,center=True)/np.sqrt(100)
print(mfcc.shape)

mpl.rcParams['axes.unicode_minus'] = False
plt.rc('font',family='NanumGothic')
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=fs,)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()