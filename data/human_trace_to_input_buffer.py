import numpy as np

def generate_address_buffer(address_list_filename, address_buffer_filename):

	input_file = open(address_list_filename, "r")
    output_file = open(address_buffer_filename, "wb")
    for line in input_file:
    	address_val = (line.strip()).astype(np.uint64)
    	output_file.write(address_val.tobytes())
    
    close(input_file)
    close(output_file)


def generate_delta_list(address_list_filename, delta_list_filename):
	input_file = open(address_list_filename, "r")
    output_file = open(address_buffer_filename, "w")
    prev_address_val = None
    for line in input_file:
    	# think that conversion will work...
    	address_val = int(line.strip())
    	if not prev_address_val:
    		delta = 0
    	else:
    		delta = address_val - prev_address_val
    	prev_address_val = address_val
    	output_file.write(str(delta) + "\n")
    close(input_file)
    close(output_file)

## will collect most frequent n_classes - 1, deltas and assign them to one-hot indices
def generate_delta_mappings_list(delta_list_filename, delta_mappings_list_filename, n_classes):
	input_file = open(delta_list_filename, "r")
	output_file = open(delta_mappings_list_filename, "w")
	# not efficient but whatever for offline training...
	d = {}
	for line in input_file:
		delta_val = int(line.strip())
		if delta_val not in d:
			d[delta_val] = 1
		else:
			d[delta_val] += 1	 
	counts = [(k, v) for key, value in d.items()]
	counts.sort(reverse=True, key = lambda x: x[1])
	### only create mappings for most frequent (n_classes - 1) deltas 
	for i in range(n_classes - 1):
		key = counts[i][0]
		val = i
		output_file.write(str(key) + "$" + str(val))

	close(input_file)
	close(output_file)


# assume input file is lines of key$value
def generate_delta_mappings(delta_mappings_list_filename, delta_mappings_buffer_filename):
	input_file = open(delta_mappings_list_filename, "r")
    output_file = open(delta_mappings_buffer_filename, "wb")
    for line in input_file:
    	key_val_split = line.strip().split("$")
    	key = key_val_split[0].astype(np.int64)
    	val = key_val_split[1].astype(np.int64)
    	output_file.write(key.tobytes())
    	output_file.write(val.tobytes())
    close(input_file)
    close(output_file)