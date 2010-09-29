#include <stdlib.h>
#include <stdio.h>

#define matrixSize 2160

main()
{
float a[matrixSize][matrixSize],
      b[matrixSize][matrixSize],
      c[matrixSize][matrixSize];
int i,j,k;
float error = 0;

/*** Initialize according to Dr. Lee's Formulae ***/
for (i=0;i<matrixSize;i++)
    for (j=0;j<matrixSize;j++)
        a[i][j]=((i+1)*(j+1))/matrixSize;

for (i=0;i<matrixSize;i++)
    for (j=0;j<matrixSize;j++)
        b[i][j]=(j+1)/(i+1);

/*** Matrix Multiplication ***/
for (i=0;i<matrixSize;i++)
    for (j=0;j<matrixSize;j++)
        for (k=0;k<matrixSize;k++)
            c[i][j] += a[i][k] * b[k][j];

/*** Error calculation ***/
for (i=0;i<matrixSize;i++)
    for (j=0;j<matrixSize;j++)
        error += abs(c[i][j]-(i+1)*(j+1));
}
