import numpy as np
import librosa
import time

starttime=time.time()
# mp3 파일 로드
music_sample=('Crossing!','Dreaming!','Flyers!!!','GlowMap','Harmony4You','Rainbow','UNION!!')
index=np.load(f"d:/study_data/_project/_data/musicindex.npy")
musics=(music_sample[i] for i in index)
concated_audio_signal=[]
concated_rfftx=[]
concated_ifftx=[]
concated_mfcc=[]
ys=[]
for i in musics:
    startload=time.time()
    audio_signal=np.load(f"d:/study_data/_project/_data/audio_signal/{i}.npy")
    concated_audio_signal.append(audio_signal)
    concated_rfftx.append(np.load(f"d:/study_data/_project/_data/r_fftx/{i}r_fftx.npy"))
    concated_ifftx.append(np.load(f"d:/study_data/_project/_data/i_fftx/{i}i_fftx.npy"))
    concated_mfcc.append(np.load(f"d:/study_data/_project/_data/mfcc/{i}mfcc.npy"))
    sample_rate=(np.load(f"d:/study_data/_project/_data/sr/{i}sr.npy"))
    ys.append(np.load(f"d:/study_data/_project/_data/ys/{i}ys.npy"))
    print(f'{i}노래 모양 :{audio_signal.shape}')
    print(f'{i}로드 시간 :{time.time()-startload}') 
concated_audio_signal=np.concatenate(concated_audio_signal,axis=0)
print(f'audio signal 모양 : {(*concated_audio_signal.shape,1)}')
concated_rfftx=np.concatenate(concated_rfftx,axis=0)
print(f'r_fftx모양 : {concated_rfftx.shape}')
concated_ifftx=np.concatenate(concated_ifftx,axis=0)
print(f'i_fftx모양 : {concated_rfftx.shape}')
concated_mfcc=np.concatenate(concated_mfcc,axis=0)
print(f'mfcc모양 : {concated_mfcc.shape}')
ys=np.concatenate(ys,axis=0)
np.save(f"d:/study_data/_project/_data/audio_signal.npy",concated_audio_signal)
np.save(f"d:/study_data/_project/_data/r_fftx.npy",concated_rfftx)
np.save(f"d:/study_data/_project/_data/i_fftx.npy",concated_ifftx)
np.save(f"d:/study_data/_project/_data/mfcc.npy",concated_mfcc)
np.save(f"d:/study_data/_project/_data/sr.npy",sample_rate)
np.save(f"d:/study_data/_project/_data/ys.npy",ys)
print(f'concate완료! sr: {sample_rate}')
print(f'runtime : {time.time()-starttime}')