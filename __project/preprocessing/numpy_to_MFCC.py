import numpy as np
import librosa
import time

def file_path(file_extension, filenum, folder=''):
    return f"d:/study_data/_project/_data/{folder}/{filenum}.{file_extension}"

index = np.load(file_path("npy", "musicindex"))
musics = ("Crossing!", "Dreaming!", "Flyers!!!", "GlowMap", "Harmony4You", "Rainbow", "UNION!!")
idolnum = np.load(file_path("npy", "idolnum"))

starttime = time.time()

for folder in musics:
    for filenum in idolnum:
        audio_signal = np.load(file_path("npy", filenum, folder))
        sample_rate = np.load(file_path("npy", f"{filenum}sr", folder))[0]
        freq = np.fft.fftfreq(len(audio_signal), 1 / sample_rate)
        mfcc = np.array([librosa.feature.mfcc(y=audio_signal[i], sr=sample_rate, n_mfcc=20, n_fft=sample_rate // 50, 
                                              hop_length=sample_rate // 200, center=True, n_mels=20,
                                              win_length=sample_rate // 50).T for i in range(len(audio_signal))])
        np.save(file_path("npy", f"{filenum}mfcc", folder), mfcc.astype(np.float32))
        
        print(f"{folder}폴더 {filenum}번 save완료! sr: {sample_rate}")
        print(f"runtime : {time.time()-starttime}")
        print(mfcc.shape)
