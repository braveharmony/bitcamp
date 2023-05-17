import tensorflow as tf
import traceback
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
if not tf.compat.v1.executing_eagerly():
    tf.compat.v1.enable_eager_execution()
print(f"텐서플로 버전 : {tf.__version__}")
print(f'즉시 실행모드 : {tf.executing_eagerly()}')
gpus=tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_visible_devices(gpus[0],'GPU')
    print(gpus)
except Exception as es:
    traceback.print_exc()
