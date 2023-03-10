def hello():
    print('Hello, world!')
hello()
print("=======================================================")
def add(a,b):
    """
    긴줄로 된 독스트링
    .
    .
    """
    print(a+b)
add(10,20)
print("=======================================================")
def add_sub(a,b):
    return a+b,a-b
x=add_sub(10,20)
print(x)
x,y=add_sub(10,20)
print(x)
print(hello.MRO())