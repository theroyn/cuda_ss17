main: ../main.cu ../helper.cu ../helper.h ../Imager.h Makefile
	nvcc -o main ../main.cu ../helper.cu ../Imager.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

