import random

def gen_mtx(N):
    mtx=[[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(N):
            num=random.randint(1000,10000)
            mtx[i][j]=num
    return mtx

def print_mydat(N,mtx):
    myfile=open("data/test/test3.in","w")
    myfile.write("{} {}".format(N,N))

    for i in range(N):
        myfile.write("\n{} ".format(i))
        for j in range(N):
            #num=random.randint(1,1000)
            num=mtx[i][j]
            myfile.write("{} ".format(num))
    myfile.close()


def print_glpdat(N,mtx):

    myfile=open("data/test/test3.dat","w")
    myfile.write("data;\n")
    myfile.write("set I:=");
    for i in range(N):
        myfile.write(str(i)+" ")
    myfile.write(";\nset J:=");
    for i in range(N): myfile.write(str(i)+" ")
    myfile.write(";\nparam c:");
    for i in range(N): myfile.write(str(i)+" ")
    myfile.write(":=")

    for i in range(N):
        myfile.write("\n{} ".format(i))
        for j in range(N):
            #num=random.randint(1,1000)
            num=mtx[i][j]
            myfile.write("{} ".format(num))

    myfile.write(";\nend;")
    myfile.close()


N=2000
mtx=gen_mtx(N)
print_mydat(N,mtx)
print_glpdat(N,mtx)

# Run glpsol:
#    glpsol --math scripts/glpk.mod --data data/test/test3.dat 