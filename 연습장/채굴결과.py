class CustomClass:
    def __len__(self):
        return 0
    def __bool__(self):
        return False
    def __str__(self):
        return ""
custom_instance = CustomClass()

if custom_instance:
    print("이 코드 블록이 실행됩니다.")
else:
    print("이 코드 블록은 실행되지 않습니다.")