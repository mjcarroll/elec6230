#include <stdio.h>
#include "mpi.h"

#define matrixSize 2160 

main(int argc, char* argv[])
{
    int my_rank, p; 
    int source, dest;
    int tag=0;
    int i, j, k, n, m, rc; 
    double start, finish;
    MPI_Status status;
    
    /* Initialize the MPI pool */
    rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &p);
    if (rc !=0)
        printf("Error initializing MPI\n");
    
    int L = 5;    
    int windowSize = 2*L + 1;
    
    /* Create the window on every PE */
    float window[windowSize][windowSize];
    for(i=0;i<windowSize;i++){
        for(j=0;j<windowSize;j++){
            window[i][j]=1.0/(windowSize*windowSize);
    }}

    /* Instantiate an offset variable */
    if(my_rank == 0)
    {
        /* Generate the matrix in thread rank 0 */
        float a[matrixSize+2*L][matrixSize+2*L];
        float result[matrixSize][matrixSize];

        for(i=0;i<matrixSize+2*L;i++){
            for(j=0;j<matrixSize+2*L;j++){
                if(i < L || j < L || i >= matrixSize+L || j >= matrixSize + L)
                    a[i][j]=0;
                else
                    a[i][j] = (j-L)%2?9:0;
        }}
        start = MPI_Wtime();
        /* Distribute the data to the other PEs */
        for(i=1; i<p; i++){
            dest = i;
            MPI_Send(&a[i*matrixSize/p][0],
                    (matrixSize+2*L)*(matrixSize/p+2*L),
                    MPI_FLOAT,
                    dest,
                    tag, 
                    MPI_COMM_WORLD);
        }
        /* Perform the convolution for this PE (rank 0) */
        for(i=0;i<matrixSize/p;i++){
            for(j=0;j<matrixSize;j++){
                result[i][j]=0;
                for(m=-L;m<=L;m++){
                    for(n=-L;n<=L;n++){
                        result[i][j] += a[i+L+m][j+L+n] * window[L+m][L+n];
        }}}}
        /* Recollect the data */
        for(i=1; i<p; i++){
            source = i;
            MPI_Recv(&result[i*matrixSize/p][0],
                    matrixSize*matrixSize/p,
                    MPI_FLOAT,
                    source,
                    tag,
                    MPI_COMM_WORLD,
                    &status);
        }

        finish = MPI_Wtime(); 
        //printf("Number of PEs: %d\n",p);
        FILE *fp;
        fp = fopen("results","a");
        fprintf(fp,"%e,",finish-start); 
        fclose(fp);
        /*FILE *fp;
        fp = fopen("results","w");
        for(i=0;i<matrixSize;i++){
            for(j=0;j<matrixSize;j++){
                fprintf(fp,"%4.6f,",result[i][j]);
            }
            fprintf(fp,"\n");
        }
        fclose(fp);*/
    }
    else
    {
        /* Initialize the data structures for this PE */ 
        int rows = matrixSize/p + 2*L;
        int cols = matrixSize + 2*L;
        float a[rows][cols];
        float result[matrixSize/p][matrixSize];

        /* Get the data for this PE */
        source = 0;
        MPI_Recv(&a,
                rows*cols,
                MPI_FLOAT,
                source,
                tag,
                MPI_COMM_WORLD,
                &status);

        /* Do the convolution */
        for(i=0;i<matrixSize/p;i++){
            for(j=0;j<matrixSize;j++){
                result[i][j]=0;
                for(m=-L;m<=L;m++){
                    for(n=-L;n<=L;n++){
                        result[i][j] += a[i+L+m][j+L+n] * window[L+m][L+n];
        }}}}

        /* Send the data back */
        dest = 0;
        MPI_Send(&result,
                matrixSize*matrixSize/p,
                MPI_FLOAT,
                dest,
                tag,
                MPI_COMM_WORLD);
    }
    MPI_Finalize();
}
