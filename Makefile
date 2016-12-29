
a.out: main.cpp
	g++ -std=c++14 -O3 -march=native -mtune=native main.cpp -lCGAL -lCGAL_Core -lgmp
