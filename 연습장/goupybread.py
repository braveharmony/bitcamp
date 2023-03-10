t=int(input())
finans=''
for tc in range(1,t+1):
    n,m,k=map(int,input().split())
    ctm=list(map(int,input().split()))
    ctm.sort()
    ans="possible"
    for arcus in range(n):
        ctm[arcus]>m*((arcus)//k)
        ans="impossible"
    finans+=f'#{tc} {ans}\n'
print(finans)