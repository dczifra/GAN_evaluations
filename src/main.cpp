
// ==================================================================
//                 EXAMPLE USAGE OF THE HUNGARIAN METHOD
// ==================================================================
// Description:
//    --> Shows the usage, and geneartes example data
//        for Hungarian method. Such as:
//        * Ramdom points from circles
//        * Random points from lines
//        * MNIST data
// Dependencies:
//    --> g++ -g -std=c++17 -o bin/main src/main.cpp -lstdc++fs
//    --> bin/main
// Author:
//    --> Domonkos Czifra, ELTE (Budapest, Hungary), 2019
//

// ==================================================================
//                           TOY DATASETS
// ==================================================================
#include <string>
#include <tuple>
#include <sstream>
#include <experimental/filesystem>

#include "hungarian.h"
#include "geom_object.h"
#include "problems.h"
#include "mincost_maxflow.tpp"

template class Flow<int>;
template class Flow<double>;

using namespace std;

/**
 * Description:
 *     Returns the perfect matching between the two picture dataset
 * Parameters:
 *     N:                        size of the train/test dataset
 *     size:                     size of the images
 *     train_folder/test_folder: the folder, where the datasets are
 */  
double match_mnist(int N)
{
    Hungarian_method method = Hungarian_method();
    //std::cout << "The matching between the pictures:\n";
    double result = method.run("/tmp/mnist_mtx.txt", "data/mnist_result2.txt");
    return result/N;
}

struct Menu
{
    bool show_help = false;
    string folder1 = "data/mnist/train";
    string folder2 = "data/mnist/test";
    pair<int, int> size = {28, 228};
    int N = 10;
    int range = -1;
    string out = "NO_OUTFILE_IS_GIVEN";
    bool deficit=false;
    bool flow=false;
};

Menu *help(vector<string> argv)
{
    Menu *m = new Menu();
    for (int i = 0; i < argv.size(); i++)
    {
        //cout<<(string) argv[i]<<endl;
        if ((string)argv[i] == "-h" || (string)argv[i] == "--help")
        {
            cout << "Description:\n"
                 << "   Min cost matching of the two picture sets\n"
                 << "usage: main [-h] [-size] [-folder1] [-folder2] [-N] [-range] [-out] \n\n"
                 << "Example: ./bin/main -size 28,28 -folder1 data/mnist/train -folder2 data/mnist/test -N 10\n"
                 << "Compile with: g++ -g -std=c++17 -o bin/main src/main.cpp -lstdc++fs \n";
            m->show_help = true;
        }
        else if ((string)argv[i] == "--ini")
        {
            fstream inifile(argv[++i]);
            argv.resize(0);

            string temp;
            while (inifile>>temp)
            {
                argv.push_back(temp);
                //inifile >> temp;
            }
            return help(argv);
        }
        else if ((string)argv[i] == "-folder1")
        {
            m->folder1 = argv[++i];
        }
        else if ((string)argv[i] == "-folder2")
        {
            m->folder2 = argv[++i];
        }
        else if ((string)argv[i] == "-size")
        {
            stringstream ss(argv[++i]);
            string first, second;
            getline(ss, first, ',');
            getline(ss, second, ',');
            m->size = {stoi(first), stoi(second)};
        }
        else if ((string)argv[i] == "-N")
        {
            m->N = stoi(argv[++i]);
        }
        else if ((string)argv[i] == "-range")
        {
            m->range = stoi(argv[++i]);
        }
        else if ((string)argv[i] == "-out")
        {
            m->out = argv[++i];
        }
        else if ((string)argv[i] == "-deficit")
        {
            m->deficit = true;
        }
        else if ((string)argv[i] == "-flow")
        {
            m->flow = true;
        }
    }
    return m;
}

double deficit(int i){
    Hungarian_method method = Hungarian_method();
    method.read_mtx("/tmp/mnist_mtx.txt");
    method.init();

    method.sparse_all(i);
    auto p=method.Gamma();
    return ((double) p.first/p.second)*100.0;
}

double flowMatching(int N){
    vector<vector<double> > mtx;
    beolv<double>(mtx,"/tmp/mnist_mtx.txt",N);
    Flow<double> f;
    return f.minCost_maxMatching_flow(mtx)/N;
}

// ====================================================================
//                                  MAIN
// ====================================================================
int main(int argc, char *argv[])
{
    //flowMatching(10);
    cout << "===== MIN COST MATCHING =====" << endl;
    vector<string> argv_;
    ofstream myfile;

    for (int i = 0; i < argc; i++)
        argv_.push_back((string)argv[i]);
    
    Menu *m = help(argv_);
    if (m->show_help)
        return 1;
    else if(m->deficit){
        generate_graph(m->N,m->size, m->folder1, m->folder2);

        myfile.open(m->out);
        myfile<<1<<" "<<100<<" "<<1<<endl; // [begin end range_by]

        for(int i=1;i<100;i++){
            cout<<"\r"<<i;
            std::cout.flush();
            myfile << deficit(i )<<" ";
        }
        cout<<endl;
    }
    else if (m->range > 0)
    {
        myfile.open(m->out);
        myfile<<m->range<<" "<<m->N<<" "<<m->range<<endl;
        
        for (int i = 1; m->range * i <= m->N; i++)
        {
            generate_graph(m->range*i,m->size, m->folder1, m->folder2);
            if(m->flow) myfile<<flowMatching(m->range*i)<<" ";
            else myfile << match_mnist(m->range*i) << " ";
        }
    }
    else
    {
        generate_graph(m->N,m->size, m->folder1, m->folder2);
        match_mnist(m->N);
    }

    myfile.close();
    delete m;


    //cout<<"===== Min Cost Perf Matching with Mincostflow ====="<<endl;
    

    return 0;
}

// ====================================================================
//                         TRASH AND UNUSED CODES
// ====================================================================
void test()
{
    pair<int, int> size(2, 2);
    Picture picture1(size);
    Picture picture2(size);

    Koord k1(1, -1);
    Koord k2(2, -1);
    Koord k3(3, -1);
    Koord k4(4, -1);

    k1.index = 0;
    k2.index = 1;
    k3.index = 2;
    k4.index = 3;

    picture1.add_point(k1);
    picture1.add_point(k2);
    picture1.add_point(k3);
    picture1.add_point(k4);
    picture2.add_point(k1);
    picture2.add_point(k2);
    picture2.add_point(k3);
    picture2.add_point(k1);

    Geometric_object::print_mtx(picture1, picture2, "data/trashexample1.txt");
    Geometric_object::print_dat(picture1, picture2, "data/trashexample1.dat");
}