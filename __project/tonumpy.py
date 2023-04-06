import numpy as np
import librosa
import time

starttime=time.time()
# mp3 파일 로드
musics=('Dreaming!','Crossing!','UNION!!','Flyers!!!','GlowMap','Harmony4You','Rainbow')
for folder in musics:
    for filenum in range(1,53):
        path = f"c:/musicforproject/{folder}_soloVer/{filenum}.mp3"
        audio_signal, sample_rate = librosa.load(path, sr=None, mono=True)

        def split_to_5sec(audio_signal,sample_rate):
            return np.array(list(audio_signal[5*i*sample_rate:5*(i+1)*sample_rate] for i in range(len(audio_signal)//(5*sample_rate))))

        audio_signal=split_to_5sec(audio_signal, sample_rate)

        # numpy 배열로 변환된 음성 신호를 파일로 저장
        np.save(f"d:/study_data/_project/_data/{folder}/{filenum}.npy", audio_signal)
        np.save(f"d:/study_data/_project/_data/{folder}/{filenum}sr.npy", np.array([sample_rate]))
        print(f'{folder}폴더 {filenum}번 save완료! sr: {sample_rate}')
        print(f'runtime : {time.time()-starttime}')
    print(audio_signal.shape)