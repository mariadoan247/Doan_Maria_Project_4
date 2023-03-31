
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = convolution_CUDA
OBJ	        = convolution_CUDA.o

default: $(EXE)

run: 
	make clean
	make
	./convolution_CUDA 2048 2048 "/home/doan0016/test_data/Problem_3/mat_input.csv" result_prob3_serial.csv time_prob3_serial.csv

convolution_CUDA.o: convolution_CUDA.cu kernel.cu
	$(NVCC) -c -o $@ convolution_CUDA.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
