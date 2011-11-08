#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

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
    fp = fopen("proj10F_openmp_out.bin","w");
    if (fp==NULL) {fputs ("File error",stderr); exit(1);}

    unsigned char *buffer = (unsigned char*) malloc(sizeof(unsigned char)*imageLength);

    for (unsigned int i = 0; i < imageLength; i++)
        buffer[i] = static_cast<unsigned char>(image[i]);

    fwrite(buffer,1,imageLength,fp);
    
    fclose(fp);

    free(buffer);
}


main(int argc, char *argv[])
{
    int numOfThreads = 1;
    
    double w0, w1;

    unsigned int imageLength = imageSize * imageSize;
    unsigned int windowLength = windowSize * windowSize;

    // Use malloc, because it tends to produce more consistent results.
    float* image = (float*) malloc(sizeof(float) * imageLength);
    float* window = (float*) malloc(sizeof(float) * windowLength);
    float* imagebuffer = (float*) malloc(sizeof(float) * imageLength);

    // Set up the convolution window.
    float windowValue = 1.0/(windowSize * windowSize);
    float windowWeight = windowSize * windowSize * windowValue;

    for(int i = 0; i < windowLength; i++)
        window[i] = windowValue;

    readFile(image,imageLength);

    if (argc==2)
        numOfThreads = atoi(argv[1]);

    omp_set_num_threads(numOfThreads);
    FILE *fp;
    fp = fopen("results.omp","a");
        
    fprintf(fp,"Number of PEs: %d\n",numOfThreads);

    w0 = omp_get_wtime();
    for (int iteration = 0; iteration < 2; iteration++)
    {
#pragma omp parallel for 
    for(int i=0; i<imageSize;i++){
        for(int j=0;j<imageSize;j++){
            float accum = 0;
            for(int m=-5;m<=5;m++){
                for(int n=-5;n<=5;n++){
                    if((i+m)<0 || (i+m)>=imageSize ||(j+n)<0 || (j+n)>=imageSize)
                        accum += 0;
                    else
                        accum += image[(i+m) * imageSize + (j + n)] *
                            window[(5 + m) * windowSize + (5 + n)];
                }
            }
           imagebuffer[(i*imageSize) + j] = accum; 
        }
    }
    image = imagebuffer;
    w1 = omp_get_wtime();
    fprintf(fp,"Iteration: %d,\tTime: %6.2f\n",iteration,w1-w0); 
    }

    writeFile(image, imageLength);

}
