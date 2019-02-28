#include <iostream>
#include "hungarian.h"
#include "problems.h"
#include "mincost_maxflow.tpp"

using namespace std;

void test1(){
    Hungarian_method method = Hungarian_method();
    method.read_mtx("data/test/test1.in");
    method.init();

    method.sparse_all(2);
    auto p=method.Gamma();
    cout<<p.first<<" "<<p.second<<" "<<((double) p.first/p.second)*100.0<<endl;
}
int main(){
    Hungarian_method method = Hungarian_method();
    method.run("data/test/test2.in","data/test/test2.out");

    vector<vector<double> > mtx;
    beolv<double>(mtx,"data/test/test2.in",4);
    Flow<double> f;
    auto p=f.minCost_maxMatching_flow(mtx);
    cout<<p.first<<" "<<p.second<<endl;
    std::cout<<"===== END ====="<<endl;
    return 0;
}