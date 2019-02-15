#include <iostream>
#include "hungarian.h"

using namespace std;

int main(){
    Hungarian_method method = Hungarian_method();
    method.read_mtx("data/test/test1.in");
    method.init();

    method.sparse_all(2);
    auto p=method.Gamma();
    cout<<p.first<<" "<<p.second<<" "<<((double) p.first/p.second)*100.0<<endl;
    std::cout<<"===== END ====="<<endl;
    return 0;
}