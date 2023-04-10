import numpy as np
import librosa
import time
import os
def file_path(file_extension, filenum, folder=''):
    return f"d:/study_data/_project/_data/{folder}/{filenum}.{file_extension}"

musics=('Crossing!','Dreaming!','Flyers!!!','GlowMap','Harmony4You','Rainbow','UNION!!')

idolnum = np.load(file_path("npy", "idolnum"))

starttime = time.time()

for folder in musics:
    for filenum in idolnum:
        audio_signal = np.load(file_path("npy", filenum, folder))
        sample_rate = np.load(file_path("npy", f"{filenum}sr", folder))[0]
        
        freq = np.fft.fftfreq(len(audio_signal), 1 / sample_rate)
        fftx = np.fft.fft(audio_signal)
        
        r_fftx = np.real(fftx) / np.sqrt(len(audio_signal))
        i_fftx = np.imag(fftx) / np.sqrt(len(audio_signal))
        r_fftx = r_fftx[:, :audio_signal.shape[1] // 2]
        r_fftx = r_fftx.reshape(*r_fftx.shape, 1)
        i_fftx = i_fftx[:, :audio_signal.shape[1] // 2]
        i_fftx = i_fftx.reshape(*i_fftx.shape, 1)
        np.save(file_path("npy", f"{filenum}r_fftx", folder), r_fftx.astype(np.float32))
        np.save(file_path("npy", f"{filenum}i_fftx", folder), i_fftx.astype(np.float32))
        print(f"{folder}폴더 {filenum}번 save완료! sr: {sample_rate}")
        print(f"runtime: {time.time() - starttime}")
