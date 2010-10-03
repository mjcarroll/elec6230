#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define matrixSize 2160 

main(int argc, char *argv[])
{
float a[matrixSize][matrixSize],
      b[matrixSize][matrixSize],
      c[matrixSize][matrixSize];
int i,j,k;
float error = 0;
int numOfThreads = 1;
int chunk = 10;
double w0,w1,w2,w3,setupInt,multInt,totalInt,errorInt;

w0 = omp_get_wtime();
printf("Parallel Implementation\n");
if (argc == 2)
    numOfThreads = atoi(argv[1]);
else if(argc == 3) 
    numOfThreads = atoi(argv[1]);
    chunk = atoi(argv[2]);

printf("Number of Threads: %d\n",numOfThreads); 
printf("Static Scheduling, Chunk Size: %d\n",chunk);

omp_set_num_threads(numOfThreads);

/*** Initialize according to Dr. Lee's Formulae ***/
#pragma omp parallel shared(a,b,c,chunk,error) private(i,j,k)
{
#pragma omp for schedule(static,chunk)
    for (i=0;i<matrixSize;i++)
        for (j=0;j<matrixSize;j++)
            a[i][j]=((i+1.0)*(j+1.0))/matrixSize;

#pragma omp for schedule(static,chunk)
    for (i=0;i<matrixSize;i++)
        for (j=0;j<matrixSize;j++)
            b[i][j]=(j+1.0)/(i+1.0);

#pragma omp for schedule(static,chunk) 
    for (i=0;i<matrixSize;i++)
        for (j=0;j<matrixSize;j++)
            c[i][j]=0;

    w1 = omp_get_wtime();
    /*** Matrix Multiplication ***/
#pragma omp for schedule(static,chunk) 
    for (i=0;i<matrixSize;i++)          // Row
        for (j=0;j<matrixSize;j++)      // Column
            for (k=0;k<matrixSize;k++)
                c[i][j] += a[i][k] * b[k][j];

    w2 = omp_get_wtime();
    /*** Error calculation ***/
#pragma omp for reduction(+:error)
    for (i=0;i<matrixSize;i++)
        for (j=0;j<matrixSize;j++)
            error += abs(c[i][j]-(i+1.0)*(j+1.0));
}

w3 = omp_get_wtime();
setupInt = w1 - w0;
multInt = w2 - w1; 
totalInt = w3 - w0;
errorInt = w3 - w2;
printf("Total Execution Time:\t%e\n",totalInt);
printf("Matrix Creation Time:\t%e\n",setupInt);
printf("Multiplication Time:\t%e\n",multInt);
printf("Error Checking Time:\t%e\n",errorInt);
printf("error = %f\n",error);
}
