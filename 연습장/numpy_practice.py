import numpy as np
import matplotlib.pyplot as plt

t=np.linspace(-10,10,1000,endpoint=False)
x=np.sin(2*np.pi*0*t)
for freq in range(1,10):
    x=x+np.sin(2*np.pi*freq*t)#-2*np.cos(2*np.pi*freq*t)
fftx=np.fft.fftshift(np.fft.fft(x))
print(fftx[:10])
r_fftx=np.real(fftx)
i_fftx=np.imag(fftx)
print(r_fftx[:10])
print(i_fftx[:10])



plt.subplot(2,2,1)
plt.plot(t,x)
plt.title('원본 함수')
plt.subplot(2,2,2)
plt.plot(freqt,r_fftx)
plt.title('리얼 함수')
plt.subplot(2,2,3)
plt.plot(freqt,i_fftx)
plt.title('이메지너리 함수')
plt.show()