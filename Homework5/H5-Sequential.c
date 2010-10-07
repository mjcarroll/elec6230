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
struct timeval t0,t1,t2,t3;

gettimeofday(&t0,0);

/*** Initialize according to Dr. Lee's Formulae ***/
for (i=0;i<matrixSize;i++)
    for (j=0;j<matrixSize;j++)
        a[i][j]=((i+1.0)*(j+1.0))/matrixSize;

for (i=0;i<matrixSize;i++)
    for (j=0;j<matrixSize;j++)
        b[i][j]=(j+1.0)/(i+1.0);

for (i=0;i<matrixSize;i++)
    for (j=0;j<matrixSize;j++)
        c[i][j]=0;

gettimeofday(&t1,0);
/*** Matrix Multiplication ***/
for (i=0;i<matrixSize;i++)          // Row
    for (j=0;j<matrixSize;j++)      // Column
        for (k=0;k<matrixSize;k++)
            c[i][j] += a[i][k] * b[k][j];
gettimeofday(&t2,0);
/*** Error calculation ***/
for (i=0;i<matrixSize;i++)
    for (j=0;j<matrixSize;j++)
        error += abs(c[i][j]-(i+1.0)*(j+1.0));
gettimeofday(&t3,0);

float totalInt = t3.tv_sec - t0.tv_sec + (t3.tv_usec - t0.tv_usec)*1.0E-06;
float setupInt = t1.tv_sec - t0.tv_sec + (t1.tv_usec - t0.tv_usec)*1.0E-06;
float multInt = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)*1.0E-06;
printf("Total Execution Time:\t%e\n",totalInt);
printf("Setup Time:\t%e\n",setupInt);
printf("Multiplication Time:\t%e\n",multInt);
printf("error = %f\n",error);
}
