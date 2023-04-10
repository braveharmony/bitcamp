
import os

musics=('Crossing!','Dreaming!','Flyers!!!','GlowMap','Harmony4You','Rainbow','UNION!!')


for folder in musics:
    save_directory = f"D:/musicforproject/{folder}_MRremove"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f'{folder}폴더 생성 완료!')
