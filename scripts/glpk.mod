set I;
set J;

param c{i in I, j in J};
var match{i in I, j in J} binary;

minimize obj: sum{j in J,i in I}match[i,j]*c[i,j];

s.t. supply{i in I}: sum{j in J}match[i,j]=1;
supply1{j in J}: sum{i in I}match[i,j]=1;


solve;


param f, symbolic := "data/test/test3.out";
printf "" >f;
for{i in I} {
    for{j in J: match[i,j]>0}{
        printf "%d %d\n",i,j>>f;
        printf "%s\n",c[i,j];
    }
}
display obj;


