vowel=['a','e','i','o','u']
voi=[3,'congratulation','synthetic','fluid']
t=voi[0]
finans=''
for testcase in range(1,t+1):
    word=voi[testcase]
    for i in vowel:
        word=word.replace(i,'')
    finans+=f'#{testcase} {word}\n'
print(finans)