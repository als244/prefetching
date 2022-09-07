from human_trace_to_input_buffer import generate_address_buffer, generate_delta_list, generate_delta_mappings_list, generate_delta_mappings

# original trace file...

BENCH_TRACE_DIR = "./mind_traces/tensorflow/"

human_trace_filename = BENCH_TRACE_DIR + "tflow1_addr.txt"
address_buffer_filename = BENCH_TRACE_DIR + "tflow1_addr.buffer"

# generate address buffer
generate_address_buffer(human_trace_filename, address_buffer_filename)

# create delta list
delta_list_filename = BENCH_TRACE_DIR + "intermediate/tflow1_deltas.txt"
generate_delta_list(human_trace_filename, delta_list_filename)

# create delta mappings in human form
N_CLASSES = 2000
delta_mappings_list_filename = BENCH_TRACE_DIR + "intermediate/tflow1_delta_mappings.txt"
generate_delta_mappings_list(delta_list_filename, delta_mappings_list_filename, N_CLASSES)

# create delta mappings buffer to be read by C model

# ASSUMING THAT EVERY APPLICATION/TRACE WILLL USE SAME MAPPINGS
# not like how it was trained before, and maybe unrealistic, but makes system sense...
delta_to_index_buffer_filename = "./mind_traces/delta_to_index.buffer"

# overrwriting system delta to index mapping with this trace...
generate_delta_mappings(delta_mappings_list_filename, delta_to_index_buffer_filename)