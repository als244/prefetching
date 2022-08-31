CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -O2 -Wall

all: lstm_prefetcher

lstm_prefetcher: lstm_prefetcher.o hash_table.o
	${CC} ${CFLAGS} -o $@ $^ -lm

lstm_prefetcher.o: lstm_prefetcher.c
	${CC} ${CFLAGS} -c lstm_prefetcher.c

# hash_table_test: hash_table.o
# 	${CC} ${CFLAGS} -o $@ $^

hash_table.o: hash_table.c
	${CC} ${CFLAGS} -c hash_table.c

clean:
	rm *.o