from sklearn.svm import LinearSVC
x1=[0,1]
x2=[0,1]
x=[[i,j]for i in x1 for j in x2]
y_and=[i and j for i in x1 for j in x2]
y_or=[i or j for i in x1 for j in x2]
y_xor=[i ^ j for i in x1 for j in x2]
print(x,y_and,y_or,y_xor)

def do_svc(x,y):
    model=LinearSVC()
    model.fit(x,y)
    print(model.score(x,y))

print('and문제')
do_svc(x,y_and)
print('or문제')
do_svc(x,y_or)
print('xor문제')
do_svc(x,y_xor)


