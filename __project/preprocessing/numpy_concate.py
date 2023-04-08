import numpy as np
import librosa
import time

def file_path(file_extension, filenum, folder=''):
    return f"d:/study_data/_project/_data/{folder}/{filenum}.{file_extension}"

starttime = time.time()
music_sample = ('Crossing!', 'Dreaming!', 'Flyers!!!', 'GlowMap', 'Harmony4You', 'Rainbow', 'UNION!!')
index = np.load(file_path("npy", "musicindex"))
musics = (music_sample[i] for i in index)
idolnum = np.load(file_path("npy", "idolnum"))
print('idolnum :', idolnum)

for folder in musics:
    concated_audio_signal = []
    concated_rfftx = []
    concated_ifftx = []
    concated_mfcc = []
    ys = []

    for filenum in idolnum:
        startload = time.time()
        audio_signal = np.load(file_path("npy", filenum, folder))
        concated_audio_signal.append(audio_signal)
        concated_rfftx.append(np.load(file_path("npy", f"{filenum}r_fftx", folder)))
        concated_ifftx.append(np.load(file_path("npy", f"{filenum}i_fftx", folder)))
        concated_mfcc.append(np.load(file_path("npy", f"{filenum}mfcc", folder)))
        sample_rate = np.load(file_path("npy", f"{filenum}sr", folder))

        y = np.zeros((len(audio_signal), 52))
        y[:, filenum - 1] = 1
        ys.append(y)

        print(f'{folder}폴더 {filenum}번 노래 모양 :{audio_signal.shape}')
        print(f'{folder}폴더 {filenum}번 로드 시간 :{time.time() - startload}')
        print(f'sample_rate :{sample_rate}')

    concated_audio_signal = np.concatenate(concated_audio_signal, axis=0)
    print(f'audio signal 모양 : {(*concated_audio_signal.shape, 1)}')
    concated_rfftx = np.concatenate(concated_rfftx, axis=0)
    print(f'r_fftx모양 : {concated_rfftx.shape}')
    concated_ifftx = np.concatenate(concated_ifftx, axis=0)
    print(f'i_fftx모양 : {concated_rfftx.shape}')
    concated_mfcc = np.concatenate(concated_mfcc, axis=0)
    print(f'mfcc모양 : {concated_mfcc.shape}')
    ys = np.concatenate(ys, axis=0)
    print(f'target 모양 :{ys.shape}')
    
    np.save(file_path("npy", f"{folder}", "audio_signal"),np.reshape(concated_audio_signal,(*concated_audio_signal.shape,1)))
    np.save(file_path("npy", f"{folder}r_fftx", "r_fftx"), concated_rfftx)
    np.save(file_path("npy", f"{folder}i_fftx", "i_fftx"), concated_ifftx)
    np.save(file_path("npy", f"{folder}mfcc", "mfcc"), concated_mfcc)
    np.save(file_path("npy", f"{folder}sr", "sr"), sample_rate)
    np.save(("npy", f"{folder}sr", "ys"),ys)
    print(f'{folder}폴더 concate완료! sr: {sample_rate}')
    print(f'runtime : {time.time()-starttime}')
