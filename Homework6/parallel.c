#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define matrixSize 2160 

main(int argc, char *argv[])
{
float a[matrixSize][matrixSize],a_new[matrixSize][matrixSize];
int i,j,m,n;
int numOfThreads = 1;
double w0,w1,w2,setupInt,totalInt,convInt;

/* Made a kernel matrix.  For this assignment it won't really matter*/
float kernel[3][3] = {{1,1,1},{1,1,1},{1,1,1}};
float kernelweight = 9;
float accum;

/* Create a matrix with entry 9, if j is even, 0 if odd */
if (argc==2)
    numOfThreads = atoi(argv[1]);

omp_set_num_threads(numOfThreads);

w0 = omp_get_wtime();

#pragma omp parallel for
for (i=0;i<matrixSize;i++){
    for (j=0;j<matrixSize;j++){
       a[i][j] =j%2?9:0;
    }
}

w1 = omp_get_wtime();

#pragma omp parallel for 
for(i=0; i<matrixSize;i++){
    for(j=0;j<matrixSize;j++){
        accum = 0;
        for(m=-1;m<=1;m++){
            for(n=-1;n<=1;n++){
                /* Handle boundary conditions:
                 * An alternative approach would be to pad the entire outside
                 * of the matrix with zeros.  I chose to use more computational
                 * time than memory for this program */
                if((i+m)<0 || (i+m)>=matrixSize ||(j+n)<0 || (j+n)>=matrixSize)
                    accum += 0;
                else
                    accum += a[i+m][j+n];
            }
        }
        a_new[i][j]=accum/kernelweight;
    }
}
/* Set the old matrix equal to the new matrix, as per the project
 * requirementsi
 */
**a = **a_new;

w2 = omp_get_wtime();

totalInt = w2 - w0;
setupInt = w1 - w0;
convInt = w2 - w1;

//printf("Total Execution Time:\t%e\n",totalInt);
//printf("Setup Time:\t\t%e\n",setupInt);
printf("%e\n",convInt);
}
