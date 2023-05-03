T=int(input())
data=[]
a=[]
b=[]
for _ in range(T):
    data.append(int(input()))
if T==1:
    print(data[0])
else:
    a.append(data[0])
    b.append(0)
    a.append(data[1])
    b.append(data[1]+a[0])
    for i in range(2,len(data)):
        a.append(max(a[i-2]+data[i],b[i-2]+data[i]))
        b.append(data[i]+a[i-1])
    print(max(a[T-1],b[T-1]))
