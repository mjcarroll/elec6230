#include <stdlib.h>
#include <stdio.h>

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

int writeFile(float* image, unsigned int imageLength){
    FILE *fp;
    fp = fopen("proj10F_sequential_out.bin","w");
    if (fp==NULL) {fputs ("File error",stderr); exit(1);}

    unsigned char *buffer = (unsigned char*) malloc(sizeof(unsigned char)*imageLength);

    for (unsigned int i = 0; i < imageLength; i++)
        buffer[i] = static_cast<unsigned char>(image[i]);

    fwrite(buffer,1,imageLength,fp);
    
    fclose(fp);

    free(buffer);
}

int main(){
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
    {
        window[i] = windowValue;
    }

    readFile(image, imageLength);

    for(int iteration = 0; iteration < 2; iteration++)
    {
        for(unsigned int i=0; i < imageSize; i++){
            for(unsigned int j=0; j < imageSize; j++){
                float accum = 0;
                for(int m=-5;m<=5;m++){
                    for(int n=-5;n<=5;n++){
                        if((i+m)<0 || (i+m) >= imageSize ||(j+n)<0||(j+n)>=imageSize)
                            accum += 0;
                        else
                            accum += image[(i+m) * imageSize + (j + n)] * 
                                     window[(5 + m) * windowSize + (5+n)];
                    }
                }
                imagebuffer[(i * imageSize) + j] = accum;
            }
        }
       
        image = imagebuffer;
    }

    writeFile(image, imageLength);

}
