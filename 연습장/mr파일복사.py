import os
import shutil
musics=('Crossing!','Dreaming!','Flyers!!!','GlowMap','Harmony4You','Rainbow','UNION!!')


for folder in musics:
    save_directory = f"D:/musicforproject/{folder}_MRremove"
    for i in range(1, 53):
        source_folder = f'{save_directory}/{i}/{i}'
        destination_folder = save_directory
        source_file = os.path.join(source_folder, f'vocals.wav')
        destination_file = os.path.join(destination_folder, f'{i}.wav')
        
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
            print(f'Copied {source_file} to {destination_file}')
        else:
            print(f'{source_file} not found')