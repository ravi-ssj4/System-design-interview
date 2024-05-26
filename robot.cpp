#include<bits/stdc++.h>
using namespace std;

int getMinCost(vector<int>& garbage, int m){
    int n = garbage.size();
    vector<vector<long>> dp(n+1, vector<long>(n+1,0));
    for(int i=n;i>=0;i--){
        for(int j=0;j<=n;j++){
            if(i==n){
                dp[i][j]=0;
            } 
            else if(j==0){
                dp[i][j] = INT_MAX;
            }
            else{
                long min_cost = INT_MAX;
                for(int k=i+1;k<=n;k++){
                    long cost = m + dp[k][j-1];
                    for(int l=i+1;l<k;l++){
                        cost += (l-i)*garbage[l];
                    }
                    min_cost = min(min_cost, cost);
                }
                dp[i][j] = min_cost;
            }
        }
    }
    long ans = INT_MAX;
    for(int j=0;j<=n;j++){
        ans = min(ans, dp[0][j]);
    }
    return ans;
}

int main(){
    vector<int> garbage = {3,4,7,3,2,2};
    int m = 10;
    cout<<getMinCost(garbage, m);
}