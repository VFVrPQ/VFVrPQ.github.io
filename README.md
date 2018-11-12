## ACM -training

### 1.Wannafly挑战赛28 - msc的背包

链接: [msc的背包](https://ac.nowcoder.com/acm/contest/217/D) 
描述： 
msc是个可爱的小女生，她喜欢许许多多稀奇古怪的小玩意。
一天，msc得到了一个大小为k的背包，她决定去买东西。
商店里有n种大小为1的物品和m种大小为2的物品。
由于msc希望买的东西尽量多，所以msc不希望买完东西之后背包还有空位（即买的所有东西的体积和必须等于k）。
她想知道自己有多少种购买物品的方案。
两种方案不同当且仅当存在一种物品在两种方案中的数量不同。
解法: 枚举有多少个1是奇数，把他们减1，他们就变成了大小为2的情况。 
代码： 
```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 3e6+10;
const int M = 998244353;

long long f[N],rf[N];
long long g[N],rg[N];
long long RP(long long a,long long b){
    long long ans=1;
    for (;b;b>>=1){
        if (b&1) ans=ans*a%M;
        a=a*a%M;
    }
    return ans;
}
long long C(int n,int m){
    return f[n]*rf[m]%M*rf[n-m]%M;
}
long long C2(int n,int m,int L){
    return g[n-L]*RP(g[n-m-L],M-2)%M*rf[m]%M;
}
int main(){
    int NN = 2e6+10;
    f[0]=1;for (int i=1;i<NN;i++) f[i]=f[i-1]*i%M;
    rf[0]=1;for (int i=1;i<NN;i++) rf[i]=RP(f[i],M-2);
    long long n,m,k;
    cin>>n>>m>>k;
    
    long long L=max(0LL,(k-n)/2);
    g[0]=max(1LL,L);///!!取0的时候0!是1，注意哦
    for (int i=1;i<N;i++) g[i]=g[i-1]*(L+i)%M;
    //rg[0]=RP(L,M-2);for (int i=1;i<N;i++)rg[i]=RP(g[i],M-2);
    
    long long ans=0;
    for (int i=0;i<=n;i++)if (i%2==k%2){
        if (k-i<0) break;
        ans+=C(n,i)*C2(n+m-1+(k-i)/2,n+m-1,L);
        ans%=M;
    }
    cout<<ans;
    return 0;
}
```

### 2. 2018NEERC - I

链接：[I. Privatization of Roads in Berland](http://codeforces.com/problemset/problem/1070/I) 
描述：n个点m条无向边，现在有100500个颜色，每个颜色最多只能涂2条边。每条边只能涂一种颜色且必须涂一种颜色.和每个点相邻的边最多只能涂k个颜色。 
问边的涂色方案是否存在，存在则输出方案。n,m,k<=600 
解法：仅有度数大于k的点会影响方案的存在，考虑度数大于k的点，degree-k超过部分必须有两条边俩俩配对，边的配对顺序无影响。考虑网络流，X部是边点（把边看承担），Y部是点。 S->边点，流量为1；边点->两端点，流量为1；点若degree>k, 点->T,流量为2（degree-k).若点到T的流量满流则有方案。 
代码： 
```c++
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <iomanip>
using namespace std;
#define DEBUG(x) cout<<x<<endl;
const int N = 2e3+10;
const int M = 1e9+7;
const int E = 10000;
const int INF = 1e9;
struct DINIC{
    int vet[E],len[E],nxt[E],head[E],cao[E];
    int d[N],vd[N],mark[N];
    int num,S,T;
    void init(int n){
        num=-1;
        memset(head,-1,sizeof head);
        S=n+1;T=n+2;
    }
    void add(int u,int v,int L)
    {
        num++;vet[num]=v;len[num]=cao[num]=L;nxt[num]=head[u];head[u]=num;
        num++;vet[num]=u;len[num]=cao[num]=0;nxt[num]=head[v];head[v]=num;
    }
    queue<int>q;
    bool BFS()
    {
        memset(d,-1,sizeof d);
        int u;
        d[S]=0;q.push(S);
        while (!q.empty())
        {
            u=q.front();q.pop();
            for (int e=head[u];e!=-1;e=nxt[e])
                if (d[vet[e]]==-1&&len[e])
                {
                    q.push(vet[e]);
                    d[vet[e]]=d[u]+1;
                }
        }
        return d[T]>=0;
    }
    int dfs(int u,int flow)
    {
        mark[u]=1;
        if (u==T)return flow;
        for (int e=head[u];e!=-1;e=nxt[e])
            if (d[vet[e]]==d[u]+1&&len[e]&&!mark[vet[e]])
            {
                int tmp=dfs(vet[e],min(flow,len[e]));
                if (tmp)
                {
                    len[e]-=tmp;
                    len[e^1]+=tmp;
                    return tmp;
                }
            }
        return 0;
    }
    int solve()
    {
        int ans=0;
        while (BFS())
        {
            memset(mark,0,sizeof mark);
            int flow=dfs(S,INF);
            if (!flow)break;
            ans+=flow;
        }
        return ans;
    }
}dinic;

struct Edge{
    int u;
    int v;
    Edge(){}
    Edge(int u,int v):u(u),v(v){}
}edges[666];
int vis[666],ds[666],res[666];
int main()
{
    //freopen("2.txt","r",stdin);
    //freopen("out.txt","w",stdout);
    int Case;scanf("%d",&Case);
    while (Case--){
        int n,m,k;scanf("%d%d%d",&n,&m,&k);
        for (int i=1;i<=n;i++)ds[i]=0;
        for (int i=1;i<=m;i++){
            int u,v;scanf("%d%d",&u,&v);
            edges[i]=Edge(u,v);
            ds[u]++;ds[v]++;
        }
        dinic.init(n+m);
        for (int i=1;i<=m;i++){
            dinic.add(i,m+edges[i].u,1);
            dinic.add(i,m+edges[i].v,1);
        }
        for (int i=1;i<=m;i++) dinic.add(dinic.S,i,1);
        int totflow=0;
        for (int i=1;i<=n;i++)if (ds[i]>k){
            dinic.add(i+m,dinic.T,2*(ds[i]-k));
            totflow+=2*(ds[i]-k);
        }
        int ans = dinic.solve();
        if (ans!=totflow){
            for (int i=1;i<m;i++)printf("0 ");printf("0\n");
            continue;
        }
        //printf("%d %d\n",ans,totflow);
        for (int i=1;i<=n;i++) vis[i]=0;
        for (int i=1;i<=m;i++) res[i]=0;
        int col=0;
        for (int i=0;i<=dinic.num && i<4*m;i+=2)if (dinic.len[i]==0){
            int u=dinic.vet[i]-m;
            //printf("%d %d\n",i,u);
            if (vis[u]){
                col++; res[(vis[u]-1)/4+1]=col;res[i/4+1]=col;
                //printf("%d %d\n",vis[u]/4+1,i/4+1);
                vis[u]=0;
            }else{
                vis[u]=i+1;
            }
        }
        for (int i=1;i<=m;i++)if (res[i]==0){
            col++;res[i]=col;
        }
        for (int i=1;i<m;i++)printf("%d ",res[i]);printf("%d\n",res[m]);
    }
    return 0;
}
```

### 3. SEERC2018 - E

链接：[Getting Deals Done](http://codeforces.com/problemset/problem/1070/E) 
描述：n个依次需要完成任务，每次完成m个任务后需要休息完成这m个任务所需要的时间，现在告诉你工作时间t，问最多能完成多少个任务，以及给出此时的阈值d。他只会去做时间<=d的任务。（n<=2e5,m<=2e5,t<=4e10) 
解法：枚举阈值，将新的阈值<=d的加入，二分能完成多少组m个任务，再二分还能完成多少个任务，可以用线段树维护，O(n\*logn\*logn) 
代码： 
```c++
/* ***********************************************
Author        : VFVrPQ
Created Time  : 五 11/ 9 15:09:20 2018
File Name     : E.cpp
Problem       : 
Description   : 
Solution      : 
Tag           : 
************************************************ */

#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <iomanip>

using namespace std;
#define DEBUG(x) cout<<x<<endl;
const int N = 2e5+10;
const int M = 1e9+7;

struct Node{
	int val;
	int id;
}a[N];
struct Tree{
	int sz[N<<2];
	long long T[N<<2];
	void build(int p,int L,int R){
		sz[p]=0;T[p]=0;
		if (L==R) return ;
		int mid=(L+R)>>1;
		build(p<<1,L,mid);
		build(p<<1|1,mid+1,R);
	}
	void ins(int p,int L,int R,int pos,int val){
		sz[p]++;T[p]+=val;
		if (L==R)return ;
		int mid=(L+R)>>1;
		if (pos<=mid) return ins(p<<1,L,mid,pos,val);
		return ins(p<<1|1,mid+1,R,pos,val);
	}
	long long find(int p,int L,int R,int bound){
		if (sz[p]<=bound) return T[p];
		if (L==R) return 0;
		int mid=(L+R)>>1;
		if (bound<=sz[p<<1]) return find(p<<1,L,mid,bound);
		auto tmp = find(p<<1,L,mid,sz[p<<1]);
		auto tmp2 = find(p<<1|1,mid+1,R,bound-sz[p<<1]);
		return tmp+tmp2;
	}
}tr;
int cmp(Node a,Node b){
	return a.val<b.val;
}
int n,m;long long t;
int pd(int sz, int bs, long long jq){
	long long tmp = tr.find(1,1,n,sz);
	return tmp*bs+jq<=t ;
}
int main()
{
	//freopen("1.txt","r",stdin);
	//freopen("out.txt","w",stdout);
	int Case;
	scanf("%d",&Case);
	while (Case--){
		scanf("%d%d%I64d",&n,&m,&t);
		int maxa=0;
		for (int i=1;i<=n;i++){
			scanf("%d",&a[i].val);
			a[i].id=i;
		}
		sort(a+1,a+n+1,cmp);
		tr.build(1,1,n);

		int ans=0,id=t;
		for (int i=1;i<=n;){
			int j=i;
			for (;j<=n;j++){
				if (a[j].val!=a[i].val)break;
				tr.ins(1,1,n,a[j].id,a[j].val);
			}
			i=j;
			int L=0,R=(i-1)/m,tt;
			while (L<=R){
				int mid=(L+R)>>1;
				if (pd(mid*m,2,0)) tt=mid, L=mid+1;
				else R=mid-1;
			}
			long long tmp = tr.find(1,1,n,tt*m);
			L=tt*m+1, R=min((tt+1)*m,i-1);int tt2=tt*m;
			while (L<=R){
				int mid=(L+R)>>1;
				if (pd(mid,1,tmp)) tt2=mid, L=mid+1;
				else R=mid-1;
			}
			if (tt2>ans){
				ans=tt2;id=a[i-1].val;
			}
		}
		printf("%d %d\n",ans,id);
	}
    return 0;
}
```

### 4. bzoj3110 整体二分 + 树状数组区间修改区间查询

链接：[bzoj3110](https://www.lydsy.com/JudgeOnline/problem.php?id=3110)
描述：有N个位置，M个操作。操作有两种，每次操作如果是1 a b c的形式表示在第a个位置到第b个位置，每个位置加入一个数c；如果是2 a b c形式，表示询问从第a个位置到第b个位置，第C大的数是多少。 
N,M<=50000,a<=b<=N,1操作中abs(c)<=N,2操作中c<=Maxlongint 
题解：对答案进行二分， 对于1操作，将c>mid的放入右区间递归，c<=mid的放入左区间； 对于2操作，若查询[mid+1,r]的数的个数>=c的，放入右区间，否则放入左区间。 
代码： 
```c++
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <iomanip>
using namespace std;
#define DEBUG(x) cout<<x<<endl;
const int N = 1e5+10;
const int M = 1e9+7;
 
struct Tree{
    long long T[2][N];
    void rst(int x,int n){
        for (;x<=n;x+=x&(-x))T[0][x]=T[1][x]=0;
    }
    void add(int x,int n,int val){
        long long val2=(long long)val*(n-x+1);
        for (;x<=n;x+=x&(-x)){
            T[0][x]+=val;
            T[1][x]+=val2;
        }
    }
    long long sum(int x,int n){
        long long bs = n-x,ret=0;
        for (;x;x-=x&(-x)) ret+=T[1][x]-T[0][x]*bs;
        return ret;
    }
}tr;
 
int n,m;
int res[N];
int t[N],a[N],b[N],c[N];
int f[N],f1[N],f2[N];
int Qh,Q[N*2];
void dfs(int L,int R,int p1,int p2){
    if (p1>p2) return ;
    if (L==R){
        for (int i=p1;i<=p2;i++)if (t[f[i]]==2) res[f[i]]=L;
        return ;
    }
    //tr.init(n);
    int mid=(L+R)>>1;
    int h1=0,h2=0;
    for (int i=p1;i<=p2;i++)if (t[f[i]]==1){
        if (c[f[i]]>mid){
            tr.add(a[f[i]],n,1);
            tr.add(b[f[i]]+1,n,-1);
            f2[h2++]=f[i];
        }else{
            f1[h1++]=f[i];
        }
    }else{
        long long tmp = tr.sum(b[f[i]],n)-tr.sum(a[f[i]]-1,n);
        if (tmp>=c[f[i]]){
            f2[h2++]=f[i];
        }else{
            f1[h1++]=f[i];
            c[f[i]]-=tmp;
        }
    }
    for (int i=0;i<h2;i++) tr.rst(a[f2[i]],n),tr.rst(b[f2[i]]+1,n);
    int h=p1;
    for (int i=0;i<h1;i++) f[h]=f1[i],h++;
    for (int i=0;i<h2;i++) f[h]=f2[i],h++;
    dfs(L,mid,p1,p1+h1-1);dfs(mid+1,R,p1+h1,p2);
}
int main()
{
    //freopen("bzoj3110.txt","r",stdin);
    //freopen("out.txt","w",stdout);
    scanf("%d%d",&n,&m);
    for (int i=1;i<=m;i++){
        scanf("%d%d%d%d",&t[i],&a[i],&b[i],&c[i]);
    }
    for (int i=1;i<=m;i++) f[i]=i;
    dfs(1,n,1,m);
    for (int i=1;i<=m;i++) if (t[i]==2)printf("%d\n",res[i]);
    return 0;
}
```
