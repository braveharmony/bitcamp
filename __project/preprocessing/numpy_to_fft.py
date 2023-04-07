import numpy as np
import librosa
import time

starttime=time.time()
# mp3 파일 로드
music_sample=('Crossing!','Dreaming!','Flyers!!!','GlowMap','Harmony4You','Rainbow','UNION!!')
index=np.load(f"d:/study_data/_project/_data/musicindex.npy")
musics=(music_sample[i] for i in index)
idolnum=np.load(f"d:/study_data/_project/_data/idolnum.npy")
for folder in musics:
    for filenum in idolnum:
        audio_signal=np.load(f"d:/study_data/_project/_data/{folder}/{filenum}.npy")
        sample_rate=np.load(f"d:/study_data/_project/_data/{folder}/{filenum}sr.npy")[0]
        freq=np.fft.fftfreq(len(audio_signal),1/sample_rate)
        fftx=np.array([np.fft.fft(audio_signal[i])for i in range(len(audio_signal))])
        r_fftx=np.real(fftx)/np.sqrt(len(audio_signal))
        r_fftx=r_fftx.reshape(*r_fftx.shape,1)
        i_fftx=np.imag(fftx)/np.sqrt(len(audio_signal))
        i_fftx=i_fftx.reshape(*i_fftx.shape,1)
        np.save(f"d:/study_data/_project/_data/{folder}/{filenum}r_fftx.npy", r_fftx[:,:audio_signal.shape[1]//2].astype(np.float32))
        np.save(f"d:/study_data/_project/_data/{folder}/{filenum}i_fftx.npy",i_fftx[:,:audio_signal.shape[1]//2].astype(np.float32))
        print(f'{folder}폴더 {filenum}번 save완료! sr: {sample_rate}')
        print(f'runtime : {time.time()-starttime}')
    