# import sys

# sys.stdin = open('연습장\inputjoyable_day.txt')
t=int(input())
ans=''
for tc in range(1,t+1):
    N=int(input())
    Boats=[]
    first_day=int(input())
    Boats.append(int(input())-1)
    for i in range(N-2):
        enjoyable_day=int(input())-1
        sameBoat=False
        for i in Boats:
            if (enjoyable_day%i)==0 : sameBoat=True
        if sameBoat==False: Boats.append(enjoyable_day)
    ans+=f'#{tc} {len(Boats)}\n'
print(ans)
