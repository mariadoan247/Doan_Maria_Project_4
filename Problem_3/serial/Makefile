all: convolution_serial.c
	gcc -g -Wall -o convolution_serial convolution_serial.c

run: 
	make clean
	make
	./convolution_serial 2048 2048 "/home/doan0016/test_data/Problem 3/mat_input.csv" result_prob3_serial.csv time_prob3_serial.csv

clean:
	$(RM) convolution_serial