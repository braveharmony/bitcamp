N=int(input())
if N==1:
    print(min(map(int,input().split())))
else:
    H=[list(map(int,input().split()))for _ in range(N)]
    R=[H[0][0]]
    G=[H[0][1]]
    B=[H[0][2]]
    for i in range(1,N):
        R.append(min(G[i-1],B[i-1])+H[i][0])
        G.append(min(R[i-1],B[i-1])+H[i][1])
        B.append(min(G[i-1],R[i-1])+H[i][2])
    print(min(R[-1],G[-1],B[-1]))