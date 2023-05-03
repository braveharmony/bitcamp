N=int(input())
person_to_work=[0 for _ in range(N)]
worksheet=[]
for i in range(N):
    worksheet.append(list(map(int,input().split())))
    person_to_work[i]=worksheet[i,i]
mincost=worksheet[0,person_to_work[0]]
for i in range(1,N):
    current_min_index=i
    current_min_value=worksheet[i,person_to_work[i]]
    for j in range(i-1):
        if worksheet[j,person_to_work[i]]-worksheet[j,person_to_work[j]]+worksheet[i,person_to_work[j]]<current_min_value:
            current_min_value=worksheet[j,person_to_work[i]]-worksheet[j,person_to_work[j]]+worksheet[i,person_to_work[j]]
            current_min_index=j
    person_to_work[i]=current_min_index
    person_to_work[current_min_index]=i
    mincost[i]=mincost[i-1]+current_min_value
print(mincost[-1])