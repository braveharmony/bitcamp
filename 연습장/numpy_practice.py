import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib as mpl

t=np.linspace(-10,10,1000,endpoint=False)

freq=np.fft.fftfreq(len(t),t[1]-t[0])


x=np.sin(2*np.pi*0*t)
for f in range(1,10):
    x=x+np.sin(2*np.pi*f*t)

fftx=np.fft.fft(x)
r_fftx=np.real(fftx)/np.sqrt(len(t))
i_fftx=np.imag(fftx)/np.sqrt(len(t))

mpl.rcParams['axes.unicode_minus'] = False
plt.rc('font',family='NanumGothic')
plt.subplot(2,2,1)
plt.plot(t,x)
plt.title('원본 함수')
plt.subplot(2,2,2)
plt.plot(freq,r_fftx)
plt.title('리얼 함수')
plt.subplot(2,2,4)
plt.plot(freq,i_fftx)
plt.title('이메지너리 함수')
plt.show()