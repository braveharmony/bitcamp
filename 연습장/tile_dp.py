def mul(a,b):
    t=[[0,0]for _ in range(2)]
    t[0][0]=a[0][0]*b[0][0]+a[0][1]*b[1][0]%10007
    t[0][1]=a[0][0]*b[0][1]+a[0][1]*b[1][1]%10007
    t[1][0]=a[1][0]*b[0][0]+a[1][1]*b[1][0]%10007
    t[1][1]=a[1][0]*b[0][1]+a[1][1]*b[1][1]%10007
    return t
def pow(a,n):
    if n==1:
        return a
    elif n%2==0:
        return mul(pow(a,n//2),pow(a,n//2))
    else:
        return mul(pow(a,n//2),pow(a,n//2+1))
n=int(input())
print(pow([[1,1],[1,0]],n+1)[1][0]%10007)