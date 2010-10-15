#include <stdlib.h>
#include <stdio.h>

#define matrixSize 2160 

main()
{
float a[matrixSize][matrixSize],
  a_new[matrixSize][matrixSize];
int i,j,m,n;
float error = 0;
struct timeval t0,t1,t2,t3;

/* Made a kernel matrix.  For this assignment it won't really matter*/
float kernel[3][3] = {{1,1,1},{1,1,1},{1,1,1}};
float kernelweight = 9;
float accum;

/* Create a matrix with entry 9, if j is even, 0 if odd */
gettimeofday(&t0,0);

for (i=0;i<matrixSize;i++)
    for (j=0;j<matrixSize;j++)
       a[i][j] =j%2?9:0;

gettimeofday(&t1,0);
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
gettimeofday(&t2,0);

float totalInt = t2.tv_sec - t0.tv_sec + (t2.tv_usec - t0.tv_usec)*1.0E-06;
float setupInt = t1.tv_sec - t0.tv_sec + (t1.tv_usec - t0.tv_usec)*1.0E-06;
float convInt = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)*1.0E-06;

/*
printf("Total Execution Time:\t%e\n",totalInt);
printf("Setup Time:\t\t%e\n",setupInt);
printf("Convolution Time:\t%e\n",convInt);
*/
printf("%e\n%e\n%e\n",totalInt,setupInt,convInt);
}
