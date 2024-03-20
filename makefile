

neural_network: onlymatrix.h ../matrix_framework/matrix.h main.c
	gcc ../matrix_framework/matrix.h -o matrix
	gcc onlymatrix.h -o nn
	gcc main.c -o main

run: neural_network
	./main

clean:
	rm -f matrix nn main