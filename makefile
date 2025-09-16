CC = clang
CFLAGS = -Wall -O2 -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Accelerate

all: main

main: main.o matOps.o matFunc.o model.o
	$(CC) $(CFLAGS) -o main main.o matOps.o matFunc.o model.o $(FRAMEWORKS)

main.o: main.c matOps.h matFunc.o model.o
	$(CC) $(CFLAGS) -c main.c

matOps.o: matOps.c matOps.h
	$(CC) $(CFLAGS) -c matOps.c

matFunc.o: matFunc.c matFunc.h matOps.h
	$(CC) $(CFLAGS) -c matFunc.c

model.o: model.c matFunc.h matOps.h
	$(CC) $(CFLAGS) -c model.c

clean:
	rm -f *.o main
