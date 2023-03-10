import sys
n,m=map(int,sys.stdin.readline().split())
nums=list(map(int,sys.stdin.readline().split()))
finans=''
dp=[]
dp.append(nums[0])
for i in range(1,n):
    dp.append(dp[i-1]+nums[i])
for _ in range(m):
    a,b=map(int,sys.stdin.readline().split())
    if b>len(dp) or a-1>len(dp):
        print(f'{a} {b} {len[dp]}')
    ans=dp[b]-dp[a-1]
    finans+=f'{ans}\n'