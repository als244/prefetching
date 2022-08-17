CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -O2 -Wall

all: transformer_prefetcher

transformer_prefetcher: transformer_prefetcher.o
	${CC} ${CFLAGS} -o $@ $^

transformer_prefetcher.o: transformer_prefetcher.c
	${CC} ${CFLAGS} -c transformer_prefetcher.c

clean:
	rm *.o