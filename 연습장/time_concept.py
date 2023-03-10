t=int(input())
# t=3
# inputs=[None,('00:00:00'),('23:59:59')
#         ,('23:59:59'),('00:00:00')
#         ,('03:29:35'),('15:01:52')]
fans=""
for testcase in range(1,t+1):
    h1,m1,s1=map(int,input().split())
    h2,m2,s2=map(int,input().split())
    # h1,m1,s1=map(int,inputs[2*testcase-1].split(":"))
    # h2,m2,s2=map(int,inputs[2*testcase].split(":"))
    sans=0; mans=0; hans=0
    if s1>s2:
        m2-=1
        sans=s2-s1+60
    else: sans=s2-s1
    if m1>m2:
        h2-=1
        mans=m2-m1+60
    else: mans=m2-m1
    hans=(h2-h1+24)%24
    fans+=f"#{testcase} {'{:0>2}'.format(str(hans))}:{'{:0>2}'.format(str(mans))}:{'{:0>2}'.format(str(sans))}\n"
print(fans)