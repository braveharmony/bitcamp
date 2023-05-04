import sys

def get_prime_list(limit):
    primes = [2]
    for num in range(3, limit + 1, 2):
        is_prime = True
        for prime in primes:
            if prime * prime > num:
                break
            if num % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

primes = get_prime_list(1000000)

def goldbach(even_number):
    for prime in primes:
        if prime > even_number:
            break
        if even_number - prime in primes:
            return (prime, even_number - prime)
    return None

import sys

t=[]
a=int(sys.stdin.readline())
while a!=0:
    t.append(a)
    a=int(sys.stdin.readline())
ans=str()
for i in t:
    a,b=goldbach(t)
    ans+=f'{t} = {a} + {b}\n'
print(ans)