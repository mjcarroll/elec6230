%%
X=1:2160;
Y=1:2160;
fid=fopen('proj10F_image_noise.bin','r');
Z=fread(fid,[2160,2160],'unsigned char');
mesh(X,Y,Z)
axis([1 2160 1 2160 0 300])

%%
fid=fopen('proj10F_sequential_out.bin','r');
Z = fread(fid,[2160,2160],'unsigned char');
figure();
mesh(X,Y,Z)
axis([1 2160 1 2160 0 300])

%%
fid=fopen('proj10F_openmp_out.bin','r');
Z = fread(fid,[2160,2160],'unsigned char');
figure();
mesh(X,Y,Z)
axis([1 2160 1 2160 0 300])