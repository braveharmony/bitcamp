import numpy as np
import librosa
import time
from sklearn.preprocessing import RobustScaler
import os
starttime=time.time()
# mp3 파일 로드

def file_path(file_extension, filenum, folder=''):
    return f"d:/study_data/_project/_data/{folder}/{filenum}.{file_extension}"

sec_to_split=2
musics=('Crossing!','Dreaming!','Flyers!!!','GlowMap','Harmony4You','Rainbow','UNION!!')
idolnum = np.load(file_path("npy", "idolnum"))
MR_removever=np.load(f"d:/study_data/_project/_data/MR_removever.npy")[0]

for folder in musics:
    for filenum in idolnum:
        path = f"D:/musicforproject/{folder}_soloVer/{filenum}.mp3"
        if MR_removever==1:
            path=f"D:/musicforproject/{folder}_MRremove/{filenum}.wav"
        audio_signal, sample_rate = librosa.load(path, sr=None, mono=True)
        before_sr=sample_rate
        sample_rate=np.load(f"d:/study_data/_project/_data/sr.npy")[0]
        audio_signal = librosa.resample(audio_signal, orig_sr=before_sr, target_sr=sample_rate).reshape(-1,1)
        scaler=RobustScaler()
        audio_signal=scaler.fit_transform(audio_signal).reshape(*audio_signal.shape[:-1])
        def split_to_sec(audio_signal,sample_rate):
            return np.array(list(audio_signal[sec_to_split*i*sample_rate:sec_to_split*(i+1)*sample_rate] for i in range(10//sec_to_split,130//sec_to_split)))

        audio_signal=split_to_sec(audio_signal, sample_rate)
        y = np.zeros((len(audio_signal), 52))
        y[:, filenum - 1] = 1
        # numpy 배열로 변환된 음성 신호를 파일로 저장
        np.save(file_path("npy", filenum, folder), audio_signal)
        np.save(file_path("npy", f"{filenum}sr", folder), np.array([sample_rate]))
        np.save(file_path("npy", f"{filenum}ys", folder), y)
        print(f'{folder}폴더 {filenum}번 save완료! sr: {sample_rate}')
        print(f'runtime : {time.time()-starttime}')
    print(audio_signal.shape)