#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	const char * DELTA_FILENAME = "5000_2500.delta_buffer";
	const char * ADDRESS_FILENAME = "5000_2500.address_buffer";

	FILE * input_file = fopen(ADDRESS_FILENAME, "rb");

	if (! input_file){
		fprintf(stderr, "Error: Cannot open address history file\n");
		exit(-1);
	}
	
	// go to end of file
	fseek(input_file, 0L, SEEK_END);
	// record size
	int size = ftell(input_file);
	// reset file pointer to read into buffer
	fseek(input_file, 0L, SEEK_SET);

	size_t els = (size_t) (size / sizeof(unsigned long));
	size_t n_addresses = els;

	unsigned long * address_history = (unsigned long *) calloc(els, sizeof(unsigned long));

	long * delta_history = (long *) calloc(els, sizeof(long));

	if (!address_history || !delta_history){
		fprintf(stderr, "Error: calloc");
		exit(-1);
	}
	
	size_t n_read = fread(address_history, sizeof(unsigned long), els, input_file);
	if (n_read != els){
		fprintf(stderr, "Error: did not read input correctly\n");
		exit(-1);
	}

	fclose(input_file);

	delta_history[0] = 0;
	for (int i = 1; i < els; i++){
		delta_history[i] = address_history[i] - address_history[i - 1];
	}

	for (int i = 0; i < els; i++){
		printf("%lu\n", address_history[i]);
	}
	printf("\n\n\n");
	for (int i = 0; i < els; i++){
		printf("%ld\n", delta_history[i]);
	}
}
