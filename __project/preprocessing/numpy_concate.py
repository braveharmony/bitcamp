import numpy as np
import librosa
import time

starttime=time.time()
# mp3 파일 로드
music_sample=('Crossing!','Dreaming!','Flyers!!!','GlowMap','Harmony4You','Rainbow','UNION!!')
index=np.load(f"d:/study_data/_project/_data/musicindex.npy")
musics=(music_sample[i] for i in index)
idolnum=np.load(f"d:/study_data/_project/_data/idolnum.npy")
print('idolnum : idolnum')
for folder in musics:
    concated_audio_signal=[]
    concated_rfftx=[]
    concated_ifftx=[]
    concated_mfcc=[]
    ys=[]
    for filenum in idolnum:
        startload=time.time()
        audio_signal=np.load(f"d:/study_data/_project/_data/{folder}/{filenum}.npy")
        concated_audio_signal.append(audio_signal)
        concated_rfftx.append(np.load(f"d:/study_data/_project/_data/{folder}/{filenum}r_fftx.npy"))
        concated_ifftx.append(np.load(f"d:/study_data/_project/_data/{folder}/{filenum}i_fftx.npy"))
        concated_mfcc.append(np.load(f"d:/study_data/_project/_data/{folder}/{filenum}mfcc.npy"))
        sample_rate=np.load(f"d:/study_data/_project/_data/{folder}/{filenum}sr.npy")
        y=np.zeros((len(audio_signal),52))
        y[:,filenum-1]=1
        ys.append(y)
        print(f'{folder}폴더 {filenum}번 노래 모양 :{audio_signal.shape}')
        print(f'{folder}폴더 {filenum}번 로드 시간 :{time.time()-startload}') 
        print(f'sample_rate :{sample_rate}')
    concated_audio_signal=np.concatenate(concated_audio_signal,axis=0)
    print(f'audio signal 모양 : {(*concated_audio_signal.shape,1)}')
    concated_rfftx=np.concatenate(concated_rfftx,axis=0)
    print(f'r_fftx모양 : {concated_rfftx.shape}')
    concated_ifftx=np.concatenate(concated_ifftx,axis=0)
    print(f'i_fftx모양 : {concated_rfftx.shape}')
    concated_mfcc=np.concatenate(concated_mfcc,axis=0)
    print(f'mfcc모양 : {concated_mfcc.shape}')
    ys=np.concatenate(ys,axis=0)
    print(f'target 모양 :{ys.shape}')
    np.save(f"d:/study_data/_project/_data/audio_signal/{folder}.npy",np.reshape(concated_audio_signal,(*concated_audio_signal.shape,1)))
    np.save(f"d:/study_data/_project/_data/r_fftx/{folder}r_fftx.npy",concated_rfftx)
    np.save(f"d:/study_data/_project/_data/i_fftx/{folder}i_fftx.npy",concated_ifftx)
    np.save(f"d:/study_data/_project/_data/mfcc/{folder}mfcc.npy",concated_mfcc)
    np.save(f"d:/study_data/_project/_data/sr/{folder}sr.npy",sample_rate)
    np.save(f"d:/study_data/_project/_data/ys/{folder}ys.npy",ys)
    print(f'{folder}폴더 concate완료! sr: {sample_rate}')
    print(f'runtime : {time.time()-starttime}')