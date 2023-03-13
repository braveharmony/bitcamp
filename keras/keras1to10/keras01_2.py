import tensorflow as tf
import platform
pythonversion=platform.python_version()
print("파이썬 버전 : ",pythonversion)
print("텐서플로 버전 : ",tf.__version__)

# terminal에 pip list 를 통해 패키지 리스트 확인 가능
# terminal에 pip install을 통해 패키지 다운로드 가능