import os
import glob
from spleeter.separator import Separator
from multiprocessing import freeze_support

def separate_music_files():
    musics = ('Crossing!', 'Dreaming!', 'Flyers!!!', 'GlowMap', 'Harmony4You', 'Rainbow', 'UNION!!')
    # Spleeter 초기화 (2 스템 모델을 사용하여 보컬과 배경 음악을 분리)
    separator = Separator("spleeter:2stems")

    for music in musics:
        input_directory = f"D:/musicforproject/{music}_soloVer/"
        output_directory = f"D:/musicforproject/{music}_MRremove/"

        for root, dirs, files in os.walk(input_directory):
            for file in files:
                print(os.path.join(root, file))

        # 디렉토리의 모든 mp3 파일을 처리합니다.
        for file_path in glob.glob(os.path.join(input_directory, "*.mp3")):
            # 입력 파일의 이름을 가져옵니다.
            file_name = os.path.basename(file_path)

            # 출력 경로를 지정합니다.
            output_path = os.path.join(output_directory, os.path.splitext(file_name)[0])

            # 소스 분리를 수행합니다.
            separator.separate_to_file(file_path, output_path)

if __name__ == '__main__':
    freeze_support()
    separate_music_files()