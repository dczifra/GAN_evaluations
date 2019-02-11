set I;
set J;

param c{i in I, j in J};
var match{i in I, j in J} binary;

minimize obj: sum{j in J,i in I}match[i,j]*c[i,j];

s.t. supply{i in I}: sum{j in J}match[i,j]=1;
supply1{j in J}: sum{i in I}match[i,j]=1;


solve;
display match;
display{i in I} supply[i];
display{i in I} supply1[i];
display obj;



data;

set I:= 0 1 2;
set J:= 0 1 2;

param c:  0 1 2 :=
0 2 1 10
1 10 2 1
2 3 10 2;

end;
