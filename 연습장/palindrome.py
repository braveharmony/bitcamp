import math
def is_palindrome(num):
    num_str = str(num)
    return num_str == num_str[::-1]
# t=int(input())
t=3
ab=[None,(1,9),(10,99),(100,1000)]
for Testcase in range(1,t+1):
    # a,b=map(int,input().split())
    a,b=ab[Testcase]
    ans=0
    for i in range(a,b+1):
        if math.sqrt(i)==math.sqrt(i)//1:
            if is_palindrome(i) and is_palindrome(int(math.sqrt(i))):
                ans+=1
    print(f'#{Testcase} {ans}')