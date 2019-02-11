
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
#include "hungarian.h"
#include "geom_object.h"
#include <string>
#include <tuple>
#include <sstream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
using namespace std;

void generate_Circles(int N, bool write_to_file = false, string filename = "data/mtx_circles")
{
    srand(time(0));
    //srand (5);
    // ===== First Cicle: =====
    Circle c1(Koord(0, 0), 2);
    c1.generate(N);

    // ===== Second Cicle: =====
    Circle c2(Koord(0, 0), 1);
    c2.generate(N);

    // ===== Write to file =====
    if (write_to_file)
    {
        c1.write("data/circles01.txt");
        c2.write("data/circles02.txt");
    }

    // Generate incidence mtx:
    Geometric_object::print_mtx(c1, c2, filename + ".txt");
    Geometric_object::print_dat(c1, c2, filename + ".dat");

    // Heuristic:
    //vector<pair<int,int> > sorted=Circle::sort(c1,c2);
    //Geometric_object::write(sorted,c1, c2, "data/HEUcircles_sorted.txt");
}
void HUN_for_circles(string filename = "data/mtx_circles", string output = "data/HUNcircles_sorted.txt")
{
    Hungarian_method method = Hungarian_method();
    method.read_mtx(filename + ".txt");
    method.init();
    method.alternating_path();
    method.print_matching(output);
}

void generate_Lines(int N, bool write_to_file = false)
{
    srand(time(0));
    //srand (5);
    // ===== First Line: =====
    Line l1(Koord(0, 0), Koord(1, 1));
    l1.generate(N);

    // ===== Second Line: =====
    Line l2(Koord(0, 1), Koord(2, 4));
    l2.generate(N);

    if (write_to_file)
    {
        l1.write("data/lines01.txt");
        l2.write("data/lines02.txt");
    }
}

// ==================================================================
//                        HUN WITH PICTURES
// ==================================================================
double wasserstein_dist(vector<vector<double>> &, vector<vector<double>> &);
vector<vector<vector<double>>> train0;
vector<vector<vector<double>>> test0;

void read_picture(std::string filename, vector<vector<double>> &pict, pair<int, int> size)
{
    ifstream myfile(filename.c_str());
    int N = size.first;
    int M = size.second;

    pict.resize(N, vector<double>(M, 0.0));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            myfile >> pict[i][j];
        }
    }
}

double euclidian_dist(vector<vector<double>> &p1, vector<vector<double>> &p2)
{
    int n = 28;
    int m = 28;
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            sum += abs(p1[i][j] - p2[i][j]) * abs(p1[i][j] - p2[i][j]);
        }
    }
    return sqrt(sum);
}

double match_mnist(int N,
                   pair<int, int> &size,
                   string train_folder,
                   string test_folder)
{
    // TODO: for small dataset save into memory instead of hard disc
    std::string filename = "/tmp/mnist_mtx.txt";
    ofstream myfile(filename.c_str());
    myfile << N << " " << N << endl;

    // TODO
    int n = 28;
    int m = 28;

    vector<vector<double>> p1, p2;
    int iterator = 0;
    string path;
    for (const auto &entry : fs::directory_iterator(train_folder))
    {
        read_picture(entry.path(), p1, {n, m});
        train0.push_back(p1);
        if ((++iterator) > N)
            break;
    }
    iterator = 0;
    for (const auto &entry : fs::directory_iterator(test_folder))
    {
        read_picture(entry.path(), p2, {n, m});
        test0.push_back(p2);
        if (++iterator > N)
            break;
    }
    srand(0);
    std::random_shuffle ( train0.begin(), train0.end() );
    srand(1);
    std::random_shuffle ( test0.begin(), test0.end() );

    for (int i = 0; i < N; i++)
    {
        myfile << i << " ";
        for (int j = 0; j < N; j++)
        {
            //cout << i << " " << j << endl;
            if (i == j)
                myfile << "99999999"
                       << " ";
            else
            {
                int pict_dist = 1;
                switch (pict_dist)
                {
                case 0: // ===== Matching with Wasserstein distance between pictures =====
                    myfile << wasserstein_dist(train0[i], test0[j]) << " ";
                case 1: // ===== Matching with Euclidean distance between pictures =====
                    myfile << euclidian_dist(train0[i], test0[j]) << " ";
                }
            }
        }
        myfile << endl;
    }
    myfile.close();

    Hungarian_method method = Hungarian_method();
    //std::cout << "The matching between the pictures:\n";
    double result = method.run("/tmp/mnist_mtx.txt", "data/mnist_result2.txt");
    return result;
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
    }
    return m;
}

// ====================================================================
//                                  MAIN
// ====================================================================
int main(int argc, char *argv[])
{
    cout << "===== MIN COST MATCHING =====" << endl;
    vector<string> argv_;
    for (int i = 0; i < argc; i++)
        argv_.push_back((string)argv[i]);
    Menu *m = help(argv_);
    if (m->show_help)
        return 1;
    else if (m->range > 0)
    {
        ofstream myfile(m->out);
        myfile<<m->range<<" "<<m->N<<" "<<m->range<<endl;
        for (int i = 1; m->range * i <= m->N; i++)
        {
            myfile << (match_mnist(m->range * i, m->size, m->folder1, m->folder2)/(m->range*i));
            myfile << " ";
        }
        myfile.close();
    }
    else
    {
        match_mnist(m->N, m->size, m->folder1, m->folder2);
    }
    delete m;
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

double wasserstein_dist(vector<vector<double>> &p1, vector<vector<double>> &p2)
{
    pair<int, int> size(28, 28);

    Picture picture1(size);
    Picture picture2(size);

    vector<int> v1, v2;

    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            Koord k = Koord(p1[i][j], -1);
            k.index = i * size.first + j;
            picture1.add_point(k);
            v1.push_back(p1[i][j]);

            Koord k2 = Koord(p2[i][j], -1);
            k2.index = i * size.first + j;
            picture2.add_point(k2);
            v2.push_back(p2[i][j]);
        }
    }

    // ===== Generate incidence mtx: =====
    // TODO: not from file
    Geometric_object::print_mtx(picture1, picture2, "data/mtx_pictures.txt");
    Geometric_object::print_dat(picture1, picture2, "data/mtx_pictures.dat");

    Hungarian_method method = Hungarian_method();
    double result = method.run("data/mtx_pictures.txt", "data/HUNpictures_sorted.txt");

    return result;
}

double match_pictures(int pict1, int pict2)
{
    vector<vector<double>> p1, p2;
    read_picture("./data/mnist/train/number_" + to_string(pict1) + ".txt", p1, {28, 28});
    read_picture("./data/mnist/test/number_" + to_string(pict2) + ".txt", p2, {28, 28});

    pair<int, int> size(28, 28);

    Picture picture1(size);
    Picture picture2(size);

    vector<int> v1, v2;

    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            Koord k = Koord(p1[i][j], -1);
            k.index = i * size.first + j;
            picture1.add_point(k);
            v1.push_back(p1[i][j]);

            Koord k2 = Koord(p2[i][j], -1);
            k2.index = i * size.first + j;
            picture2.add_point(k2);
            v2.push_back(p2[i][j]);
        }
    }
    /*sort(v1.begin(),v1.end());
    sort(v2.begin(),v2.end());
    for(int i=0;i<v1.size();i++){
        cout<<v1[i]<<" "<<v2[i]<<endl;
    }*/

    // Generate incidence mtx:
    // TODO: not from file
    Geometric_object::print_mtx(picture1, picture2, "data/mtx_pictures.txt");
    Geometric_object::print_dat(picture1, picture2, "data/mtx_pictures.dat");

    Hungarian_method method = Hungarian_method();
    double result = method.run("data/mtx_pictures.txt", "data/HUNpictures_sorted.txt");

    return result;
}

/*for (int i = 0; i < N; i++)
{
    vector<vector<double>> p1, p2;
    read_picture("./data/mnist/train/number_" + to_string(i) + ".txt", p1, {n, m});
    train0.push_back(p1);

    read_picture("./data/mnist/test/number_" + to_string(i) + ".txt", p2, {n, m});
    test0.push_back(p2);
}*/