import subprocess
import numpy as np
# 실행할 파일 목록
files_to_run = ["tonumpy.py", "numpy_to_fft.py", "numpy_to_MFCC.py", "numpy_concate.py", "last_concate.py"]

sampling_rate=np.array([8000])
idolnum=np.array(range(1,53))
musicindex=np.array([0,1,2,3,4,5,6])

np.save(f"d:/study_data/_project/_data/sr.npy", sampling_rate)
np.save(f"d:/study_data/_project/_data/idolnum.npy", idolnum)
np.save(f"d:/study_data/_project/_data/musicindex.npy", musicindex)


for file in files_to_run:
    # 파일을 실행
    file=f'./__project/preprocessing/{file}'
    result = subprocess.run(["python", file], stdout=None, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"Error occurred while running {file}:")
        print(result.stderr.decode("utf-8"))
    else:
        print(f"{file} has been executed successfully.")