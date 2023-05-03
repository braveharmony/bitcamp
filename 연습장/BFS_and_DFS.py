def DFS(v,node,visited,dfs):
    visited[v]=1
    dfs.append(v)
    for i in node[v]:
        if visited[i]!=1:
            dfs=DFS(i,node,visited,dfs)
    return dfs
def BFS(v,node,visited,bfs):
    visited[v]=1
    bfs.append(v)
    queue=[v]
    while queue:
        for i in node[queue.pop(0)]:
            if visited[i]!=1:
                queue.append(i)
                bfs.append(i)
                visited[i]=1
    return bfs
N,M,V=map(int,input().split())
node=[[]for _ in range(N+1)]
for _ in range(M):
    a,b=map(int,input().split())
    node[a].append(b)
    node[b].append(a)
for i in node:
    i.sort()
dfs=DFS(V,node,[0 for _ in node],[])
bfs=BFS(V,node,[0 for _ in node],[])
print(' '.join(map(str,dfs)))
print(' '.join(map(str,bfs)))
