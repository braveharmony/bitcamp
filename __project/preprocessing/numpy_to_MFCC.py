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
        mfcc=np.array([librosa.feature.mfcc(y=audio_signal[i], sr=sample_rate, n_mfcc=sample_rate//100
                            , n_fft=sample_rate//50, hop_length=sample_rate//200,center=True,n_mels=200).T for i in range(len(audio_signal))])
        np.save(f"d:/study_data/_project/_data/{folder}/{filenum}mfcc.npy", mfcc.astype(np.float32))
        print(f'{folder}폴더 {filenum}번 save완료! sr: {sample_rate}')
        print(f'runtime : {time.time()-starttime}')
        print(mfcc.shape)
    