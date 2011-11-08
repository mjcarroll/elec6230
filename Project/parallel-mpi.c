#include <stdio.h>
#include "mpi.h"

#define matrixSize 2160 
#define windowSize 11

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
    
    /* Create the window on every PE */
    float windowValue = 1.0/(windowSize * windowSize);
    float windowWeight = windowSize * windowSize * windowValue;

    for(int i = 0; i < windowLength; i++)
        window[i] = windowValue;

    /* Instantiate an offset variable */
    if(my_rank == 0)
    {
        /* Generate the matrix in thread rank 0 */
        float* image = (float*) malloc(sizeof(float) * imageLength);
        
        readFile(image, imageLength);
        
        start = MPI_Wtime();
        /* Distribute the data to the other PEs */
        for(i=1; i<p; i++){
            dest = i;
            MPI_Send(&image[i*imageSize/p][0],
                    (imageSize+2*L)*(imageSize/p+2*L),
                    MPI_FLOAT,
                    dest,
                    tag, 
                    MPI_COMM_WORLD);
        }
        /* Perform the convolution for this PE (rank 0) */
        for(i=0;i<imageSize/p;i++){
            for(j=0;j<imageSize;j++){
                result[i][j]=0;
                for(m=-L;m<=L;m++){
                    for(n=-L;n<=L;n++){
                        result[i][j] += a[i+L+m][j+L+n] * window[L+m][L+n];
        }}}}
        /* Recollect the data */
        for(i=1; i<p; i++){
            source = i;
            MPI_Recv(&result[i*imageSize/p][0],
                    imageSize*imageSize/p,
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
