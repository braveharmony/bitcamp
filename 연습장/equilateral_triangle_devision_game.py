# voi=[2,2,1,3,3]
# t=voi[0]
t=int(input())
finans=''
for testcase in range(1,t+1):
    # a=voi[2*testcase-1];b=voi[2*testcase]
    a,b=map(int,input().split())
    ans=int(a/b)**2
    finans+=f'#{testcase} {ans}\n'
print(finans)