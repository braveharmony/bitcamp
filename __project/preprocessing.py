import subprocess
import numpy as np

#######################################################
# 52명의 캐릭터와 7종류의 곡중 어느걸 쓸것인지
# sampling_rate는 몇으로 리셈플링 할것인지 아래를 통해 결정
#######################################################
sampling_rate=np.array([8000])
idolnum=range(1,53)
musicindex=[0,1,2,3,4,5,6]
MR_removever=1
########################################################################################################

np.save(f"d:/study_data/_project/_data/sr.npy", np.array(list(sampling_rate)))
np.save(f"d:/study_data/_project/_data/idolnum.npy", np.array(list(idolnum)))
np.save(f"d:/study_data/_project/_data/musicindex.npy", np.array(list(musicindex)))
np.save(f"d:/study_data/_project/_data/MR_removever.npy", np.array([MR_removever]))

########################################################################################################

# 실행할 파일 목록
files_to_run = ["mkdir_for_preprocess.py","tonumpy.py", "numpy_to_fft.py", "numpy_to_MFCC.py","numpy_to_stft.py", "numpy_concate.py", "last_concate.py"]
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
