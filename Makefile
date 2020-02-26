
CUDA_LIBS = -lcublas
C11 = -std=c++11

# If are using UACH server use this
#LIBS = -L/usr/lib/gcc/x86_64-pc-linux-gnu/6.4.1/ -llapack -lblas -lfftw3 -lgfortran -lm
LIBS = -L/usr/lib -llapack -lblas -lfftw3 -lm

MAIN= comparing_dct.cpp
DCT_PROC= ./include/dct/cublas/DctCuBlas.cu

BUILD_DIR = ./build

O_OBJS = main.o dct_proc.o

O_OBJS2 = $(addprefix $(BUILD_DIR)/, $(O_OBJS))

all: init clean objs
	nvcc $(C11) $(LIBS) $(CUDA_LIBS) $(O_OBJS2) -o app

init:
	if [ -d $(BUILD_DIR) ]; then echo "$(BUILD_DIR) exists!"; else mkdir $(BUILD_DIR); fi

clean:
	rm -r -f ./app
	rm -r -f $(BUILD_DIR)/*

objs:
	nvcc -x cu $(C11)  $(LIBS)  -dc -o $(BUILD_DIR)/dct_proc.o $(DCT_PROC)
	nvcc -x cu $(C11)  $(LIBS)  -dc -o $(BUILD_DIR)/main.o $(MAIN)






