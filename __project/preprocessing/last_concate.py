import numpy as np
import librosa
import time

def file_path(file_extension, filenum, folder=''):
    return f"d:/study_data/_project/_data/{folder}/{filenum}.{file_extension}"


starttime=time.time()
# mp3 파일 로드
music_sample=('Crossing!','Dreaming!','Flyers!!!','GlowMap','Harmony4You','Rainbow','UNION!!')
index=np.load(f"d:/study_data/_project/_data/musicindex.npy")
musics=(music_sample[i] for i in index)
concated_audio_signal=[]
concated_rfftx=[]
concated_ifftx=[]
concated_mfcc=[]
concated_stft=[]
ys=[]
for folder in musics:
    startload=time.time()
    audio_signal = np.load(file_path('npy', folder, 'audio_signal'))
    concated_audio_signal.append(audio_signal)
    concated_rfftx.append(np.load(file_path('npy', f"{folder}r_fftx", 'r_fftx')))
    concated_ifftx.append(np.load(file_path('npy', f"{folder}i_fftx", 'i_fftx')))
    concated_mfcc.append(np.load(file_path('npy', f"{folder}mfcc", 'mfcc')))
    concated_stft.append(np.load(file_path("npy", f"{folder}stft","stft")))
    sample_rate = (np.load(file_path('npy', f"{folder}sr", 'sr')))
    ys.append(np.load(file_path("npy", f"{folder}ys", "ys")))
    print(f'{folder}노래 모양 :{audio_signal.shape}')
    print(f'{folder}로드 시간 :{time.time()-startload}') 
concated_audio_signal=np.concatenate(concated_audio_signal,axis=0)
print(f'audio signal 모양 : {(*concated_audio_signal.shape,1)}')
concated_rfftx=np.concatenate(concated_rfftx,axis=0)
print(f'r_fftx모양 : {concated_rfftx.shape}')
concated_ifftx=np.concatenate(concated_ifftx,axis=0)
print(f'i_fftx모양 : {concated_rfftx.shape}')
concated_mfcc=np.concatenate(concated_mfcc,axis=0)
print(f'mfcc모양 : {concated_mfcc.shape}')
concated_stft = np.concatenate(concated_stft, axis=0)
print(f'stft모양 : {concated_stft.shape}')
ys=np.concatenate(ys,axis=0)
np.save(file_path('npy', 'audio_signal'), concated_audio_signal)
np.save(file_path('npy', 'r_fftx'), concated_rfftx)
np.save(file_path('npy', 'i_fftx'), concated_ifftx)
np.save(file_path('npy', 'mfcc'), concated_mfcc)
np.save(file_path('npy', 'stft'), concated_stft)
np.save(file_path('npy', 'sr'), sample_rate)
np.save(file_path('npy', 'ys'), ys)
print(f'concate완료! sr: {sample_rate}')
print(f'runtime : {time.time()-starttime}')