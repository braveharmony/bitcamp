import subprocess
import numpy as np

#######################################################
# 52명의 캐릭터와 7종류의 곡중 어느걸 쓸것인지
# sampling_rate는 몇으로 리셈플링 할것인지 아래를 통해 결정
#######################################################
sampling_rate=np.array([8000])
idolnum=[1,2,3,4]
musicindex=[0,1,2,3,4,5,6]

# # 특정 음악만 뽑고 싶으면
# audio_signal=np.load(f"d:/study_data/_project/_data/audio_signal/{music}.npy")
# r_fftx=np.load(f"d:/study_data/_project/_data/r_fftx/{music}r_fftx.npy")
# i_fftx=np.load(f"d:/study_data/_project/_data/i_fftx/{music}i_fftx.npy")
# mfcc=np.load(f"d:/study_data/_project/_data/mfcc/{music}mfcc.npy")
# sr=np.load(f"d:/study_data/_project/_data/sr/{music}sr.npy")
# y=np.load(f"d:/study_data/_project/_data/ys/{music}ys.npy")


# # 모든 데이터를 뽑고 싶으면
# audio_signal=np.load(f"d:/study_data/_project/_data/audio_signal.npy")
# r_fftx=np.load(f"d:/study_data/_project/_data/r_fftx.npy")
# i_fftx=np.load(f"d:/study_data/_project/_data/i_fftx.npy")
# mfcc=np.load(f"d:/study_data/_project/_data/mfcc.npy")
# sr=np.load(f"d:/study_data/_project/_data/sr.npy")
# y=np.load(f"d:/study_data/_project/_data/ys.npy")

########################################################################################################

np.save(f"d:/study_data/_project/_data/sr.npy", np.array(list(sampling_rate)))
np.save(f"d:/study_data/_project/_data/idolnum.npy", np.array(list(idolnum)))
np.save(f"d:/study_data/_project/_data/musicindex.npy", np.array(list(musicindex)))

########################################################################################################
# 실행할 파일 목록
files_to_run = ["tonumpy.py", "numpy_to_fft.py", "numpy_to_MFCC.py", "numpy_concate.py", "last_concate.py"]
folder="./__project/preprocessing/"

for file in files_to_run:
    # 파일을 실행
    file=f'{folder}{file}'
    result = subprocess.run(["python", file], stdout=None, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"Error occurred while running {file}:")
        print(result.stderr.decode("utf-8"))
    else:
        print(f"{file} has been executed successfully.")
