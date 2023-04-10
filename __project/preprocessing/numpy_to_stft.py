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
        pad_width = int(0.5 * (sample_rate // 100))
        audio_signal = np.pad(audio_signal, ((0, 0), (pad_width, pad_width)), mode='constant')

        # Compute complex STFT
        stft_result = np.array([librosa.stft(audio_signal[i], n_fft=sample_rate // 50, 
                                               hop_length=sample_rate // 100, center=False, window='hamm',
                                               win_length=sample_rate // 50)[:sample_rate // 100] for i in range(len(audio_signal))])
        
        # Separate real and imaginary parts
        stft_real = np.real(stft_result)
        stft_imag = np.imag(stft_result)

        # Stack real and imaginary parts along a new axis
        stft_4d = np.stack((stft_real, stft_imag), axis=-1)

        np.save(file_path("npy", f"{filenum}stft", folder), stft_4d.astype(np.float32))
        
        print(f"{folder}폴더 {filenum}번 save완료! sr: {sample_rate}")
        print(f"runtime : {time.time()-starttime}")
        print(stft_4d.shape)
