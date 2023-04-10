import numpy as np
import time
import os

def file_path(file_extension, filenum, folder=''):
    return f"d:/study_data/_project/_data/{folder}/{filenum}.{file_extension}"

music_sample=('Crossing!','Dreaming!','Flyers!!!','GlowMap','Harmony4You','Rainbow','UNION!!')
index=np.load(f"d:/study_data/_project/_data/musicindex.npy")
musics=(music_sample[i] for i in index)
idolnum = np.load(file_path("npy", "idolnum"))

starttime = time.time()

for folder in musics:
    save_directory = f"d:/study_data/_project/_data/{folder}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f'{folder}폴더 생성 완료!')
        
        
folders_to_create = ["audio_signal", "r_fftx", "i_fftx", "mfcc", "stft", "sr", "ys"]
for folder in folders_to_create:
    save_directory = f"d:/study_data/_project/_data/{folder}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f'{folder}폴더 생성 완료!')
        