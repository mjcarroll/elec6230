#include <stdio.h>
#include "mpi.h"

#define imageSize 2160 
#define windowSize 11

void readFile(float* image, unsigned int imageLength){
    FILE *fp;
    fp = fopen("proj10F_image_noise.bin","r");
    if (fp==NULL) {fputs ("File error",stderr); exit(1);}

    unsigned char *buffer = (unsigned char*) malloc(sizeof(unsigned char)*imageLength);
    if (buffer == NULL) {fputs ("Memory error",stderr); exit(2);}

    fread(buffer,1,imageLength,fp);

    fclose(fp);

    for (unsigned int i = 0; i < imageLength; i++)
        image[i] = static_cast<float>(buffer[i]);

    free(buffer);
}

void writeFile(float* image, unsigned int imageLength){
    FILE *fp;
    fp = fopen("proj10F_mpi_out.bin","w");
    if (fp==NULL) {fputs ("File error",stderr); exit(1);}

    unsigned char *buffer = (unsigned char*) malloc(sizeof(unsigned char)*imageLength);

    for (unsigned int i = 0; i < imageLength; i++)
        buffer[i] = static_cast<unsigned char>(image[i]);

    fwrite(buffer,1,imageLength,fp);
    
    free(buffer);
}

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
    printf("Size: %d",p); 
    int L = 5;    
    
    unsigned int imageLength = imageSize * imageSize;
    unsigned int windowLength = windowSize * windowSize;

    /* Create the window on every PE */
    float windowValue = 1.0/(windowSize * windowSize);
    float windowWeight = windowSize * windowSize * windowValue;

    float* window = (float*) malloc(sizeof(float) * windowLength);
    
    for(int i = 0; i < windowLength; i++)
        window[i] = windowValue;

    /* Instantiate an offset variable */
    if(my_rank == 0)
    {
        FILE *fp;
        fp = fopen("results","a");
        
        fprintf(fp,"Number of PEs: %d\n",p);

        /* Generate the matrix in thread rank 0 */
        float* image = (float*) malloc(sizeof(float) * imageLength);
        float* imagebuffer = (float*) malloc(sizeof(float) * imageLength);

        readFile(image, imageLength);
        
        start = MPI_Wtime();
        
        int iterations = 0;

        /* Distribute the data to the other PEs */
        for(i=1; i<p; i++){
            dest = i;
            if(i == p - 1)
                MPI_Send(&image[(i * (imageSize/p) - L) * imageSize],
                        imageSize * (imageSize/p + L),
                        MPI_FLOAT,
                        dest,
                        tag,
                        MPI_COMM_WORLD);
            else
                MPI_Send(&image[(i * (imageSize/p) - L) * imageSize],
                        imageSize * (imageSize/p + 2*L),
                        MPI_FLOAT,
                        dest,
                        tag, 
                        MPI_COMM_WORLD);
        }
        
        /* Perform the convolution for this PE (rank 0) */
        for(i=0; i < imageSize/p; i++){
            for(j=0; j < imageSize; j++){
                imagebuffer[i*imageSize + j]=0;
                for(m=-L;m<=L;m++){
                    for(n=-L;n<=L;n++){
                        if((i+m)<0 || (j+n)<0 || (j+n)>=imageSize)
                             imagebuffer[i*imageSize + j] += 0;
                        else
                            imagebuffer[i*imageSize + j] += 
                                image[(i+m) * imageSize + (j+n)] * 
                                window[(L+m) * windowSize + (L+n)];
        }}}}

        image = imagebuffer;

        MPI_Send(&image[(L+imageSize) * imageSize],
                imageSize * L,
                MPI_FLOAT,
                my_rank + 1,
                tag,
                MPI_COMM_WORLD);
        
        MPI_Recv(&image[(imageSize - L) * imageSize],
                imageSize * L,
                MPI_FLOAT,
                my_rank - 1,
                tag,
                MPI_COMM_WORLD,
                &status);

        for(i=0; i < imageSize/p; i++){
            for(j=0; j < imageSize; j++){
                imagebuffer[i*imageSize + j]=0;
                for(m=-L;m<=L;m++){
                    for(n=-L;n<=L;n++){
                        if((i+m)<0 || (j+n)<0 || (j+n)>=imageSize)
                             imagebuffer[i*imageSize + j] += 0;
                        else
                            imagebuffer[i*imageSize + j] += 
                                image[(i+m) * imageSize + (j+n)] * 
                                window[(L+m) * windowSize + (L+n)];
        }}}}

        /* Recollect the data */
        for(i=1; i<p; i++){
            source = i;
            MPI_Recv(&imagebuffer[i*imageSize/p * imageSize],
                    imageSize*imageSize/p,
                    MPI_FLOAT,
                    source,
                    tag,
                    MPI_COMM_WORLD,
                    &status);
        }

        image = imagebuffer;
        
        free(imagebuffer);
        finish = MPI_Wtime();
        fprintf(fp,"Iteration %d: %e\n",iterations,finish-start);         

        writeFile(image, imageLength);
        fclose(fp);
    }
    else
    {
        /* Initialize the data structures for this PE */ 
        float* image = (float*) malloc(
                sizeof(float) * imageSize * imageSize/p + 2 * L);
        float* imagebuffer = (float*) malloc(
                sizeof(float) * imageSize * imageSize/p + 2 * L);
        
        /* Get the data for this PE */
        source = 0;
        if(my_rank == p - 1)
            MPI_Recv(&image[0],
                    imageSize * (imageSize/p + L),
                    MPI_FLOAT,
                    source,
                    tag,
                    MPI_COMM_WORLD,
                    &status);
        else
            MPI_Recv(&image[0],
                    imageSize * (imageSize/p + 2*L),
                    MPI_FLOAT,
                    source,
                    tag,
                    MPI_COMM_WORLD,
                    &status);

        /* Do the convolution */
        for(i=0;i<imageSize/p;i++){
            for(j=0;j<imageSize;j++){
                imagebuffer[i*imageSize + j]=0;
                for(m=-L;m<=L;m++){
                    for(n=-L;n<=L;n++){
                        if((i+m)<0 || 
                                (i+m)>= imageSize/p || 
                                (j+n)<0 || 
                                (j+n)>=imageSize)
                            imagebuffer[i*imageSize + j] += 0;
                        else
                            imagebuffer[i*imageSize + j] += 
                                image[(i+m) * imageSize + (j+n)] * 
                                window[(L+m) * windowSize + (L+n)];
        }}}}
        
        image = imagebuffer;

        /* Send top rows to my_rank-1 */
        MPI_Send(&image[L*imageSize],
                imageSize * L,
                MPI_FLOAT,
                my_rank - 1,
                tag,
                MPI_COMM_WORLD);
        if (my_rank != p - 1)
            MPI_Recv(&image[(L+imageSize) * imageSize],
                    imageSize * L,
                    MPI_FLOAT,
                    my_rank + 1,
                    tag,
                    MPI_COMM_WORLD,
                    &status);

        /* Send bottom rows to my_rank + 1 */
        if (my_rank != p - 1)
            MPI_Send(&image[(L+imageSize) * imageSize],
                    imageSize * L,
                    MPI_FLOAT,
                    my_rank + 1,
                    tag,
                    MPI_COMM_WORLD);
        
        MPI_Recv(&image[0],
                imageSize * L,
                MPI_FLOAT,
                my_rank - 1,
                tag,
                MPI_COMM_WORLD,
                &status);

        /* Convolve again */
        for(i=0;i<imageSize/p;i++){
            for(j=0;j<imageSize;j++){
                imagebuffer[i*imageSize + j]=0;
                for(m=-L;m<=L;m++){
                    for(n=-L;n<=L;n++){
                        if((i+m)<0 || 
                                (i+m)>= imageSize/p || 
                                (j+n)<0 || 
                                (j+n)>=imageSize)
                            imagebuffer[i*imageSize + j] += 0;
                        else
                            imagebuffer[i*imageSize + j] += 
                                image[(i+m) * imageSize + (j+n)] * 
                                window[(L+m) * windowSize + (L+n)];
        }}}}


        /* Send the data back */
        dest = 0;
        MPI_Send(&imagebuffer[L*imageSize],
                imageSize * imageSize/p,
                MPI_FLOAT,
                dest,
                tag,
                MPI_COMM_WORLD);
        
        free(image);
        free(imagebuffer);
    }
    MPI_Finalize();
}
