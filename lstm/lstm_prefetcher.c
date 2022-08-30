#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>

#ifndef  M_PI
#define  M_PI  3.1415926535897932384626433
#endif

#include "lstm_prefetcher.h"

/*For booking purposes in comments:
LET input dim = I, hidden dim = H, output dim = O, sequence length = S, batch size = N,
lets assume I=O=# of classes=K (such that input is one hot encoding of dim K and output is distribution over K classes)
*/

float sample_gaussian(float mean, float var) {

	if (var == 0){
		return mean;
	}
	float x = (float)rand() / RAND_MAX;
  float y = (float)rand() / RAND_MAX;
  float z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
  float std = sqrt(var);
  float val = std * z + mean;
  return val;
}

// allocating space for container structs and data buffers to hold params

// params = 4(HI + H^2 + H) + HO + O 
// space (bytes) = 4 * #params = 4 * (4(HI + H^2 + H) + HO + O)
Params * init_model_parameters(Dims model_dims, bool is_zero){

	// parameter skeletons
	Params* params = malloc(sizeof(Params));
	Embed_Weights* embed_weights = malloc(sizeof(Embed_Weights));
	Biases* biases = malloc(sizeof(Biases));
	Hidden_Weights* hidden_weights = malloc(sizeof(Hidden_Weights));
	if ((! params) || (!embed_weights) || (!biases) || (!hidden_weights)){
		fprintf(stderr, "Error: Malloc\n");
		exit(-1);
	}

	// ALLOCATE SPACE FOR PARAMTERS
	

	/* EMBEDDING WEIGHTS (from token at individual timestep to hidden dimention) */

	// LSTM CELL (these weights are multiplied by x_t)
	embed_weights -> content = malloc(model_dims.hidden * model_dims.input * sizeof(float));
	embed_weights -> remember = malloc(model_dims.hidden * model_dims.input * sizeof(float));
	embed_weights -> new_input = malloc(model_dims.hidden * model_dims.input * sizeof(float));
	embed_weights -> pass_output = malloc(model_dims.hidden * model_dims.input * sizeof(float));

	// output layer
	embed_weights -> classify = malloc(model_dims.output * model_dims.hidden * sizeof(float));

	if ((!(embed_weights->content)) || (!(embed_weights->remember)) || (!(embed_weights->new_input)) ||
			(!(embed_weights->pass_output)) || (!(embed_weights->classify))){
		fprintf(stderr, "Error: Malloc\n");
		exit(-1);
	}

	// USE A WEIGHT INITALIZATION METHOD TO FILL THE BUFFERS
	init_weights(embed_weights -> content, model_dims.hidden * model_dims.input, model_dims.input, model_dims.hidden, is_zero);
	init_weights(embed_weights -> remember, model_dims.hidden * model_dims.input, model_dims.input, model_dims.hidden, is_zero);
	init_weights(embed_weights -> new_input, model_dims.hidden * model_dims.input, model_dims.input, model_dims.hidden, is_zero);
	init_weights(embed_weights -> pass_output, model_dims.hidden * model_dims.input, model_dims.input, model_dims.hidden, is_zero);
	init_weights(embed_weights -> classify, model_dims.hidden * model_dims.input, model_dims.hidden, model_dims.output, is_zero);


	/* BIASES (added to the embedding + hidden vectors) */

	// LSTM CELL (these biases are added to Wx + Uh)
	biases -> content = calloc(model_dims.hidden, sizeof(float));
	biases -> remember = calloc(model_dims.hidden, sizeof(float));
	biases -> new_input = calloc(model_dims.hidden, sizeof(float));
	biases -> pass_output = calloc(model_dims.hidden, sizeof(float));
	
	// output layer (added to W_classify * h_final)
	biases -> classify = calloc(model_dims.output, sizeof(float));

	if ((!(biases->content)) || (!(biases->remember)) || (!(biases->new_input)) 
			|| (!(biases->pass_output)) || (!(biases->classify))){
		fprintf(stderr, "Error: Malloc\n");
		exit(-1);
	}

	// biases already initialized to 0

	//* HIDDEN WEIGHTS (applying mapping from previous hidden state to current) */

	// LSTM CELL (these weights are multiplied by h_(t-1)
	hidden_weights -> content = malloc(model_dims.hidden * model_dims.hidden * sizeof(float));
	hidden_weights -> remember = malloc(model_dims.hidden * model_dims.hidden * sizeof(float));
	hidden_weights -> new_input = malloc(model_dims.hidden * model_dims.hidden * sizeof(float));
	hidden_weights -> pass_output = malloc(model_dims.hidden * model_dims.hidden * sizeof(float));
	
	if ((!(hidden_weights->content)) || (!(hidden_weights->remember)) || (!(hidden_weights->new_input)) ||
			(!(hidden_weights->pass_output))) {
		fprintf(stderr, "Error: Malloc\n");
		exit(-1);
	}

	init_weights(hidden_weights -> content, model_dims.hidden * model_dims.hidden, model_dims.hidden, model_dims.hidden, is_zero);
	init_weights(hidden_weights -> remember, model_dims.hidden * model_dims.hidden, model_dims.hidden, model_dims.hidden, is_zero);
	init_weights(hidden_weights -> new_input, model_dims.hidden * model_dims.hidden, model_dims.hidden, model_dims.hidden, is_zero);
	init_weights(hidden_weights -> pass_output, model_dims.hidden * model_dims.hidden, model_dims.hidden, model_dims.hidden, is_zero);

	// link the data buffers for parameters to the structure
	params -> embed_weights = embed_weights;
	params -> biases = biases;
	params -> hidden_weights = hidden_weights;

	return params;
}


// HE INITIALIZATION [sample from N(0, 1/(fan_in + fan_out))]
void init_weights(float *weights, int size, int unit_inputs, int unit_outputs, bool is_zero){
	float mean = 0.0f;
	float var;
	if (is_zero){
		var = 0.0;
	}
	else{
		var = 1.0f / (unit_inputs + unit_outputs);
	}
	for (int i = 0; i < size; i++){
		weights[i] = sample_gaussian(mean, var);
	}
}

// individual cell space (bytes): 24 * HN
LSTM_Cell * init_lstm_cell(Dims model_dims, int batch_size){

	LSTM_Cell * cell = malloc(sizeof(LSTM_Cell));
	if (!cell) {
		fprintf(stderr, "Error: Malloc\n");
		exit(-1);
	}

	cell -> content_temp = calloc(model_dims.hidden * model_dims.batch_size, sizeof(float));
	cell -> content = calloc(model_dims.hidden * model_dims.batch_size, sizeof(float));
	cell -> remember = calloc(model_dims.hidden * model_dims.batch_size, sizeof(float));
	cell -> new_input = calloc(model_dims.hidden * model_dims.batch_size, sizeof(float));
	cell -> pass_output = calloc(model_dims.hidden * model_dims.batch_size, sizeof(float));
	cell -> hidden = calloc(model_dims.hidden * model_dims.batch_size, sizeof(float));

	if ( (!(cell->content_temp)) || (!(cell->content)) || (!(cell->remember)) ||
		 (!(cell->new_input)) || (!(cell->pass_output)) || (!(cell->hidden)) ) {
		fprintf(stderr, "Error: Calloc\n");
		exit(-1);
	}

	return cell;
}

// forward buffer total space (bytes): S * 24 * HN + 8 * ON
Forward_Buffer * init_forward_buffer(Dims model_dims, int batch_size){

	Forward_Buffer * buff = malloc(sizeof(Forward_Buffer));
	if (!buff) {
		fprintf(stderr, "Error: Malloc\n");
		exit(-1);
	}
	LSTM_Cell ** cells = malloc(model_dims.seq_length * sizeof(LSTM_Cell *));
	if (!cells) {
		fprintf(stderr, "Error: Malloc\n");
		exit(-1);
	}
	for (int i = 0; i < model_dims.seq_length; i++){
		cells[i] = init_lstm_cell(model_dims, batch_size);
	}

	buff -> cells = cells;
	buff -> linear_output = calloc(model_dims.output * batch_size, sizeof(float));
	buff -> label_distribution = calloc(model_dims.output * batch_size, sizeof(float));
	if ((!(buff -> linear_output)) || (!(buff -> label_distribution))){
		fprintf(stderr, "Error: Calloc\n");
		exit(-1);
	}

	return buff;
}

// backprop buff total space (bytes): 4*(4(HI + H^2 + H) + HO + O) + 24*HN
Backprop_Buffer * init_backprop_buffer(Dims model_dims, int batch_size){

	Backprop_Buffer * buff = malloc(sizeof(Backprop_Buffer));
	if (!buff) {
		fprintf(stderr, "Error: Malloc\n");
		exit(-1);
	}

	buff -> output_layer_deriv = calloc(model_dims.output * batch_size, sizeof(float));
	buff -> param_derivs = init_model_parameters(model_dims, true);
	buff -> prev_means = init_model_parameters(model_dims, true);
	buff -> prev_var = init_model_parameters(model_dims, true);
	buff -> cell_derivs = init_lstm_cell(model_dims, batch_size);

	return buff;
}


// model has (4(HI + H^2 + H) + HO + O) params
// takes up 4*(4(HI + H^2 + H) + HO + O) bytes
LSTM * init_lstm(int input_dim, int hidden_dim, int output_dim, int seq_length){

	Dims model_dims = {.input = input_dim, .hidden = hidden_dim, .output = output_dim, 
							.seq_length = seq_length };
	
	LSTM * model = malloc(sizeof(LSTM));
	if (!model){
		fprintf(stderr, "Error: Malloc\n");
		exit(-1);
	}
	
	model -> dims = model_dims;
	model -> params = init_model_parameters(model_dims, false);

	return model;
}

// during training the model takes up 40HK + 32H^2 + 32H + 8K + N*[(S+1)*24*H + 8*K]

Train_LSTM * init_trainer(LSTM * model, float learning_rate, float mean_decay, float var_decay, int batch_size, int n_epochs){

	Train_LSTM * trainer = malloc(sizeof(Train_LSTM));
	if (!trainer){
		fprintf(stderr, "Error: Malloc\n");
		exit(-1);
	}

	trainer -> model = model;
	trainer -> forward_buffer = init_forward_buffer(model -> dims, batch_size);
	trainer -> backprop_buffer = init_backprop_buffer(model -> dims, batch_size);
	trainer -> learning_rate = learning_rate;
	trainer -> mean_decay = mean_decay;
	trainer -> var_decay = var_decay;
	trainer -> batch_size = batch_size;
	trainer -> n_epochs = n_epochs;
	trainer -> loss = 0;

	return trainer;

}


long * read_raw_training_data(const char * filename, int * n_addresses, unsigned long ** address_history){

	FILE * input_file = fopen(filename, "rb");

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
	*n_addresses = els;

	// PROBLEM SPECIFIC INPUT (assume buffer of long's, then do one hot encoding afterwards)
	*address_history = (unsigned long *) calloc(els, sizeof(unsigned long));
	
	long * delta_history = (long *) calloc(els, sizeof(long));

	if (!address_history || !delta_history){
		fprintf(stderr, "Error: calloc");
		exit(-1);
	}
	
	size_t n_read = fread(address_history, sizeof(unsigned long), els, input_file);
	if (n_read != size){
		fprintf(stderr, "Error: did not read input correctly\n");
		exit(-1);
	}

	fclose(input_file);

	delta_history[0] = 0;
	for (int i = 1; i < els; i++){
		delta_history[i] = address_history[i] - address_history[i - 1];
	}

	return delta_history;
}


void add_delta_to_index_mappings(HashTable * ht, const char * filename){

	FILE * input_file = fopen(filename, "rb");

	if (!input_file){
		fprintf(stderr, "Error: Cannot open delta -> index mappings file\n");
		exit(-1);
	}

	// go to end of file
	fseek(input_file, 0L, SEEK_END);
	// record size
	int size = ftell(input_file);
	// reset file pointer to read into buffer
	fseek(input_file, 0L, SEEK_SET);

	size_t els = (size_t) (size / sizeof(long));

	long * delta_index_mappings = (long *) calloc(els, sizeof(long));

	if (! delta_index_mappings){
		fprintf(stderr, "Error: Calloc\n");
		exit(-1);
	}

	size_t n_read = fread(delta_index_mappings, sizeof(long), els, input_file);
	if (n_read != size){
		fprintf(stderr, "Error: did not read input correctly\n");
		exit(-1);
	}

	fclose(input_file);

	long delta;
	unsigned int index;
	for (int i = 0; i < els; i+=2){
		delta = delta_index_mappings[i];
		index = (unsigned int) (delta_index_mappings[i + 1]);
		ht_insert(ht, delta, index);
	}

	free(delta_index_mappings);
	return;

}


Batch * init_general_batch(Train_LSTM * trainer, unsigned int * training_data, int training_data_length){
	int batch_size = trainer -> batch_size;
	int seq_length = (trainer -> model -> dims).seq_length;
	Batch * mini_batch = (Batch * ) malloc(sizeof(Batch));
	if (!mini_batch){
		fprintf(stderr, "Error: malloc");
		exit(-1);
	}
	// starting index of encoded delta within training data for the sequence
	mini_batch -> training_ind_seq_start = (unsigned int *) calloc(batch_size, sizeof(unsigned int));
	// value of encoded delta to predict after sequence
	mini_batch -> correct_label_encoded = (unsigned int *) calloc(batch_size, sizeof(unsigned int));
	// input token for the model at each step in sequence and for every batch (rows are unique timestep, cols are values for samp in batch)
	// storing here to access the inputs to model easily in backprop
	mini_batch -> input_token_ids = (unsigned int *) calloc(seq_length * batch_size, sizeof(unsigned int));
	if ( (!(minibatch -> training_ind_seq_start)) ||  (!(minibatch -> correct_label_encoded)) || (!(minibatch -> current_input_token_ids))){
		fprintf(stderr, "Error: calloc");
		exit(-1);
	}
	mini_batch -> training_data = encoded_deltas;
	mini_batch -> training_data_length = n_addresses;
	return mini_batch;
}

void populate_batch(Train_LSTM * trainer, Batch * mini_batch){
	int batch_size = trainer -> batch_size;
	int seq_length = (trainer -> model -> dims).seq_length;
	int lower = seq_length;
	int upper = mini_batch -> training_data_length - seq_length - 1;
	for (int i = 0; i < batch_size; i++){
		 int rand_start = (rand() % (upper - lower + 1)) + lower;
		 mini_batch -> training_ind_seq_start[i] = rand_start;
		 mini_batch -> input_token_ids[i] = mini_batch -> training_data[rand_start];
		 mini_batch -> correct_label_encoded[i] = mini_batch -> training_data[rand_start + seq_length];
	}
}

// assume were are doing out += A * B (matrix mutliply but add to exiting values)
// where A = (M, K) and B = (K, N)
// for a given cell (i, j) in output we are doing <ith row of A, jth col of B>
// OVERWRITE_OUTPUT WITH MATMUL
void simp_mat_mul(float * restrict A, float * restrict B, float * restrict out, int M, int K, int N){
	memset(out, 0, M * N * sizeof(float));
	for (int row = 0; row < M; row++){
		for (int k = 0; k < K; k++){
			for (int col = 0; col < N; col++){
				out[row * N + col] += A[row * K + k] * B[k * N + col]
			}
		}
	}
}

// assume were are doing out += A * B (matrix mutliply but add to exiting values)
// where A = (M, K) and B = (K, N)
// for a given cell (i, j) in output we are doing <ith row of A, jth col of B>
// ADD MATMUL RESULT TO EXISTING MATRIX OUT
void simp_mat_mul_add(float * restrict A, float * restrict B, float * restrict out, int M, int K, int N){
	for (int row = 0; row < M; row++){
		for (int k = 0; k < K; k++){
			for (int col = 0; col < N; col++){
				out[row * N + col] += A[row * K + k] * B[k * N + col]
			}
		}
	}
}

// assume were are doing out += A * (transpose B) (matrix mutliply but add to exiting values)
// where A = (M, K) and B = (N, K)
// for a given cell (i, j) in output we are doing <ith row of A, jth row of B>
// ADD MATMUL RESULT TO EXISTING MATRIX OUT
void simp_mat_mul_right_trans(float * restrict A, float * restrict B, float * restrict out, int M, int K, int N){
	for (int row = 0; row < M; row++){
		for (int k = 0; k < K; k++){
			for (int col = 0; col < N; col++){
				out[row * N + col] += A[row * K + k] * B[col * K + k];
			}
		}
	}
}

// assume were are doing out += (transpose A) * B (matrix mutliply but add to exiting values)
// where A = (K, M) and B = (K, N)
// for a given cell (i, j) in output we are doing <ith col of A, jth col of B>
// ADD MATMUL RESULT TO EXISTING MATRIX OUT
void simp_mat_mul_left_trans(float * restrict A, float * restrict B, float * restrict out, int M, int K, int N){
	for (int row = 0; row < M; row++){
		for (int k = 0; k < K; k++){
			for (int col = 0; col < N; col++){
				out[row * N + col] += A[k * M + row] * B[k * N + col];
			}
		}
	}
}

void my_hadamard(float * restrict A, float * restrict B, float * restrict out, int size){
	for (int i = 0; i < size; i++){
		out[i] = A[i] * B[i];
	}
}

// hadamard product but add to existing values
void my_hadamard_add(float * restrict A, float * restrict B, float * restrict out, int size){
	for (int i = 0; i < size; i++){
		out[i] += A[i] * B[i];
	}
}

void my_tanh(float * restrict A, int size){
	for (int i = 0; i < size; i++){
		A[i] = tanhf(A[i]);
	}
}

void my_hadamard_right_tanh(float * restrict A, float * restrict B, float * restrict out, int size){
	for (int i = 0; i < size; i++){
		out[i] = A[i] * tanhf(B[i]);
	}
}

void my_cell_content_deriv(float * restrict A, float * restrict B, float * restrict C, float * restrict out, int size){
	for (int i = 0; i < size; i++){
		out[i] += A[i] * B[i] * (1 - (tanhf(C[i]) * tanhf(C[i])));
	}
}

void my_sigmoid_activ_deriv(float * restrict cell_state, float * restrict out, int size){
	float * val;
	for (int i = 0; i < size; i++){
		val = cell_state[i];
		out[i] = val * (1 - val);
	}
}

void my_upstream_activ_deriv(float * restrict upstream_deriv, float * restrict sigmoid_activ_deriv, float * restrict out, int size){
	for (int i = 0; i < size; i++){
		out[i] = upstream_deriv[i] * sigmiod_activ_deriv[i];
	}
}

// mimicing matrix multiplication of (upstream, transpose(input))
// adding to the exiting values of "out" which are a global gradient matrix
// add to "out" at each timestep for a given embedding matrix gradient

// at a given timestemp, for each token id in batch, add the column of batch's index in upstream matrix to the token id's column in result
// *not cache efficient, should've transposed embedding matrix to memory coalesce better...
void add_to_embed_weight_gradient(unsigned int * input_tokens, float * restrict upstream_activ_deriv_buff,  float * restrict out, int time_step, int hidden_dim, int batch_size){

	unsigned int token_id;
	for (int s = 0; s < batch_size; s++){
		// in range [0, input_dim)
		token_id = input_token[time_step * batch_size + s];
		// add column of sample in batch from upstream to the token_id column in result
		// upstream (hidden_dim, batch size)
		// result (hidden_dim, input_dim)
		for (int h = 0; h < hidden_dim; h++){
			out[h * hidden_dim + token_id] += upstream_activ_deriv_buff[h * batch_size + s];
		}
	}
}



// could figure out faster way (exp is slow)
void my_sigmoid(float * restrict A, int size){
	for (int i = 0; i < size; i++){
		A[i] = 1.0f / (1 + expf(-1 * A[i]))
	}
}

void my_softmax(float * restrict A, float * restrict out, int output_dim, int batch_size) {
  for (int s = 0; s < batch_size; s++){
	  float m = -INFINITY;
	  for (int i = 0; i < output_dim; i++) {
	    if (A[i * batch_size + s] > m) {
	      m = A[i * batch_size + s];
	    }
	  }

	  float sum = 0.0;
	  for (int i = 0; i < output_dim; i++) {
	    sum += expf(A[i * batch_size + s] - m);
	  }

	  float offset = m + logf(sum);
	  for (int i = 0; i < output_dim; i++) {
	    out[i * batch_size + s] = expf(A[i * batch_size + s] - offset);
	  }
	}
}

void forward_pass(Train_LSTM * trainer, Batch * mini_batch){

	Param * model_params = trainer -> model -> params;
	Embed_Weights * embeddings = model_params -> embed_weights;
	Biases * biases = model_params -> biases;
	Hidden_Weights * hidden_weights = model_params -> hidden_weights;

	int batch_size = trainer -> batch_size;
	Dims dims = trainer -> model -> dims;
	int seq_length = dims.seq_length;
	int hidden_dim = dims.hidden;
	int input_dim = dims.input;
	int output_dim = dims.output;
	unsigned int * training_ind_seq_start = mini_batch -> training_ind_seq_start;
	unsigned int * training_data = mini_batch -> training_data;
	unsigned int * current_input_token_ids = mini_batch -> current_input_token_ids;
	unsigned int training_ind_start;
	unsigned int input_token;

	LSTM_Cell ** cells = trainer -> forward_buffer -> cells;

	// if we really wanted could probably parallelize content_temp vs. remember vs. new_input vs. pass_output
	for (int t = 0; t < seq_length; t++){
		// advance the token id references by 1
		LSTM_Cell * cell = cells[t];
		// NOT VERY EFFICIENT (could align better with memory coalsceing for cache), BUT OK...	
		// for EMBEDDING need to get corresponding column in weight matrix as token id
		// then copy this weight column to the # in batch'th column in the LSTM cell value
		for (int k = 0; k < batch_size; k++){
			training_ind_start = training_ind_seq_start[k];
			input_token = training_data[training_ind_start + t];
			input_token_ids[t * batch_size + k] = input_token;
			// GET EMBEDDING VALUES (columns of embedding weights indexed by token id) + bias
			// write to column to cell intermediate values
			for (int h = 0; h < hidden_dim; h++){
				// current content
				cell -> content_temp[k + h * batch_size] = embeddings -> content[input_token + h * input_dim] + biases -> content[h];
				// remember
				cell -> remember[k + h * batch_size] = embeddings -> remember[input_token + h * input_dim] + biases -> remember[h];
				// new
				cell -> new_input[k + h * batch_size] = embeddings -> new_input[input_token + h * input_dim] + biases -> new_input[h];
				// output
				cell -> pass_output[k + h * batch_size] = embeddings -> pass_output[input_token + h * input_dim] + biases -> pass_output[h];
			}
		}

		// ADD HIDDEN WEIGHTS * PREVIOUS HIDDEN STATE
		if (t != 0){
			float * prev_hidden = cells[t - 1] -> hidden;
			// current content
			simp_mat_mul_add(hidden_weights -> content, prev_hidden, cell -> content_temp, hidden_dim, hidden_dim, batch_size);
			// remember
			simp_mat_mul_add(hidden_weights -> remember, prev_hidden, cell -> remember, hidden_dim, hidden_dim, batch_size);
			// new
			simp_mat_mul_add(hidden_weights -> new_input, prev_hidden, cell -> new_input, hidden_dim, hidden_dim, batch_size);
			// output
			simp_mat_mul_add(hidden_weights -> pass_output, prev_hidden, cell -> pass_output, hidden_dim, hidden_dim, batch_size);
		}

		// NON_LINEAR ACTIVATE
		int n_els = hidden_dim * batch_size;
		// current content
		my_tanh(cell->content_temp, n_els);
		// remember
		my_sigmoid(cell -> remember, n_els);
		// new
		my_sigmoid(cell -> new_input, n_els);
		// output
		my_sigmoid(cell -> pass_output, n_els);

		// GO TO NEXT STEP
		
		// content
		// remember
		if (t != 0){
			float * prev_content = cells[t - 1] -> content;
			my_hadamard_add(cell -> remember, prev_content, cell -> content, n_els);
		}
		// new input
		my_hadamard_add(cell -> new_input, cell -> content_temp, cell -> content, n_els);
		
		// hidden
		my_hadamard_right_tanh(cell -> pass_output, cell -> content, cell -> hidden, n_els);
	}
	// now we have last value for hidden and pass into linear layer
	// want to comput x_out = w_y * h_last + b_y
	float * output_hidden =  cells[seq_length - 1] -> hidden;
	float * linear_output = trainer -> forward_buffer -> linear_output;
	simp_mat_mul(embeddings -> classify, output_hidden, linear_output, output_dim, hidden_dim, batch_size);

	for (int i = 0; i < output_dim; i++){
		for (int j = 0; j < batch_size; j++){
			linear_output[i * batch_size + j] += biases -> classify[i];
		}
	}
	// perform softmax
	// (output_dim, batch_size)
	float *label_distribution = trainer -> forward_buffer -> label_distribution; 
	my_softmax(linear_output, label_distribution, output_dim, batch_size);

	// done with forward pass (output in trainer -> forward_buffer -> label_distribution)
	return;
}


void backwards_pass(Train_LSTM * trainer, Batch * mini_batch){
		
		Dims dims = trainer -> model -> dims;
		int input_dim = dims.input;
		int hidden_dim = dims.hidden;
		int output_dim = dims.output;
		int seq_length = dims.seq_length;

		int batch_size = trainer -> batch_size;

		/* CURRENT MODEL PARAMETERS */
		Params * model_params = trainer -> model -> params;
		Embed_Weights * model_embed_weights = model_params -> embed_weights;
		Biases * model_biases = model_params -> biases;
		Hidden_Weights * model_hidden_weights = model_params -> hidden_weights;

		/* STATES FROM FORWARD PASS */
		Forward_Buffer * forward_buffer = trainer -> forward_buffer;
		LSTM_Cell ** forward_cells = forward_buffer -> cells;
		unsigned int * input_token_ids = mini_batch -> input_token_ids;

		/* STATES TO POPULATE IN BACKWARD PASS */

		// general buffer for backprop
		Backprop_Buffer * backprop_buffer = trainer -> backprop_buffer;

		// parameter derivitive buffers
		Params * param_derivs = backprop_buffer -> param_derivs;
		Embed_Weights * embed_weight_derivs = param_derivs -> embed_weights;
		Biases * bias_derivs = param_derivs -> biases;
		Hidden_Weights * hidden_weight_derivs = param_derivs -> hidden_weights;

		// current cell derivative buffers (used to calculate param derivs)
		LSTM_Cell * cell_derivs = backprop_buffer -> cell_derivs;

		/* extra helper variable sfor computation */
		// will need to free at end of backprop...
		// (hidden_dim, 1) used for bias matmul generality
		float * ones_hidden_dim = (float *) malloc(hidden_dim * sizeof(float));
		memset(ones_hidden_dim, 1, hidden_dim * sizeof(float));

		float * ones_batch_dim = (float *) malloc(batch_size * sizeof(float));
		memset(ones_batch_dim, 1, batch_size * sizeof(float));
		

		/* START COMPUTING DERIVATIVES... */

		/* OUTPUT LAYER DERIVS */ 
		float * predicted = forward_buffer -> label_distribution;
		float * correct_labels = mini_batch -> correct_label_encoded;
		
		float * output_deriv = backprop_buffer -> output_layer_deriv;
		// get dL/dX_out
		// = (predicted - correct) => normalized by batch size
		// because correct is one-hot encoding we can copy predicted and subtract 1 from correct value
		memcpy(output_deriv, predicted, output_dim * batch_size * sizeof(float));
		int correct_ind;
		for (int s = 0; s < batch_size; s++){
			correct_ind = correct_labels[s];
			output_deriv[correct_ind * batch_size + s] -= 1;
		}

		// get dW_classify
		// = matmul(dL/dX_out, transpose(last hidden layer))
		float * last_hidden = forward_cells[seq_length - 1] -> hidden;
		simp_mat_mul_right_trans(output_deriv, last_hidden, embed_weight_derivs -> classify, output_dim, batch_size, hidden_dim);

		// get dB_classify
		// = matmul(dL/dX_out, ones_hidden_dim)
		simp_mat_mul(output_deriv, ones_hidden_dim, bias_derivs -> classify, output_dim, hidden_dim, 1);


		/* BRIDGE TO LSTM MODEL DERIVS... */
		int n_els = hidden_dim * batch_size;

		// get dL/d_hidden_last
		// setting cell_derivs -> hidden
		float * weight_classify = model_embed_weights -> classify;
		float * last_hidden_state_deriv = cell_derivs -> hidden;
		simp_mat_mul_left_trans(weight_classify, output_deriv, last_hidden_state_deriv, hidden_dim, output_dim, batch_size);
		// make sure that cell_derivs -> content starts fresh as 0 (might be duplicate setting of 0, could optimize...)
		memset(cell_derivs -> content, 0, n_els * sizeof(float));

		/* INTERNAL LSTM DERIVS */

		// the deriv of last hidden state is the bridge...
		// will be the same computations for every internal lstm cell*
		// *with a couple values being passed between cells

		LSTM_Cell * cur_cell, *prev_cell;

		// will be used as intermediate buffers within backprop, but will be cleared after each use
		// IF NEED BE CAN OPTIMIZE THESE...
		float * sigmoid_activ_deriv_buff = calloc(n_els, sizeof(float));
		float * upstream_activ_deriv_buff = calloc(n_els, sizeof(float));
		// will be buffer for each type of cell state used to multiply with upstream gradient and added to prev hidden deriv
		float * state_deriv_wrt_prev_hidden_buff = calloc(n_els, sizeof(float));
		// will accumulate the deriv wrt to all cell states and will pass back to next iteration
		float * prev_hidden_deriv_buff = calloc(n_els, sizeof(float));
		// will store loss deriv wrt to prev content state and will pass back to next iteration
		float * prev_content_deriv_buff = calloc(n_els, sizeof(float));

		// computing derivs until the first cell
		// what do about t = 0 case??
		for (int t = seq_length - 1; t > 0; t--){
			prev_cell = forward_cells[t - 1];
			cur_cell = forward_cells[t];

			/* GET CELL STATE DERIVS */
			// cell_derivs -> hidden is populated on first pass through the "output bridge", 
			// otherwise populated by prior iteration
			// cell_derivs -> content: starts with values from prior iteration, so will add to them. first pass starts as zeros
			// cell_derivs -> [Ctemp|R|N|O]: are over-written each iteration

			// dL/dC_t
			my_cell_content_deriv(cell_derivs -> hidden, cur_cell -> pass_output, cur_cell -> content, cell_derivs -> content, n_els);

			// dL/dO_t
			my_hadamard_right_tanh(cell_derivs -> hidden, cur_cell -> content, cell_derivs -> pass_output, n_els);

			// dL/dR_t
			my_hadamard(cell_derivs -> content, prev_cell -> content, cell_derivs -> remember, n_els);

			// dL/dN_t
			my_hadamard(cell_derivs -> content, cur_cell -> content_temp, cell_derivs -> new_input, n_els);

			// dL/dCtemp_t
			my_hadamard(cell_derivs -> content, cur_cell -> new_input, cell_derivs -> content_temp, n_els);

			/* COMPUTE PARAM DERIVS */

			// repeat for all K_t = {Ctemp_t, R_t, N_t, O_t}

			// ORGANIZED LISTS TO STORE POINTERS TO DATA
			// for easy iterations of duplicate logic
			float * cell_states_list[] = {cur_cell -> content_temp, cur_cell -> remember, cur_cell -> new_input, cur_cell -> pass_output};
			float * upstream_state_derivs_list[] = {cell_derivs -> content_temp, cell_derivs -> remember, cell_derivs -> new_input, cell_derivs -> pass_output};
			float * embed_weight_derivs_list[] = {embed_weight_derivs -> content, embed_weight_derivs -> remember, embed_weight_derivs -> new_input, embed_weight_derivs -> pass_output};
			float * bias_derivs_list[] = {bias_derivs -> content, bias_derivs -> remember, bias_derivs -> new_input, bias_derivs -> pass_output};
			float * hidden_weight_derivs_list[] = {hidden_weight_derivs -> content, hidden_weight_derivs -> remember, hidden_weight_derivs -> new_input, hidden_weight_derivs -> pass_output};
			float * model_hidden_weights_list[] = {model_hidden_weights -> content, model_hidden_weights -> remember, model_hidden_weights -> new_input, model_hidden_weights -> pass_output};

			// k takes on various states meaning [0 -> content, 1 -> remember, 2 -> input, 3 -> output]
			for (int k = 0; k < 4; k++){
				
				//hadamard(K_t, (1 - K_t))
				// over-writes sigmoid acitv deriv buff within function
				my_sigmoid_activ_deriv(cell_states_list[k], sigmoid_activ_deriv_buff, n_els);

				// get hadamard(dL/dK_t, hadamard(K_t, (1 - K_t))) which will be used for all param updates
				// over-writes upstream_activ_deriv_buff within function
				my_upstream_activ_deriv(upstream_state_derivs_list[k], sigmoid_activ_deriv_buff, upstream_activ_deriv_buff, n_els);

				// do special computation for weights because dependent on input X which is supposed to be one-hot vectors,
				// but in implementation of one-hot matrix does not exist in order to save memory
				// thus do special matrix computations (to mimic typical matmul of one hot) given our input data structure
				add_to_embed_weight_gradient(input_token_ids, upstream_activ_deriv_buff, embed_weight_derivs_list[k], t, hidden_dim, batch_size);

				// compute bias gradient
				// matmul(upstream, ones_batch_size)
				simp_mat_mul_add(upstream_activ_deriv_buff, ones_batch_dim, bias_derivs_list[k], hidden_dim, batch_size, 1);

				// compute hidden weight gradient
				// matmul(upstream, transpose(prev_hidden_state))
				simp_mat_mul_right_trans(upstream_activ_deriv_buff, prev_cell -> hidden, hidden_weight_derivs_list[k], hidden_dim, batch_size, hidden_dim);

				// compute derivate with respect to prev hidden state
				// dK_t / dh_(t-1)
				// matmaul(Hidden Weights for K_t, hadamard(K_t, (1 - K_t)))
				// these will be used to compute dL/dh_(t-1), which is bridge to next cell back (along with dL/dC_(t-1))
				// add matmul(dL/dK_t, dK_t/dh_(t-1)) to dL/dh_(t-1)
				// resets state_deriv buffer within function, then writes to it
				simp_mat_mul(model_hidden_weights_list[k], sigmoid_activ_deriv_buff, state_deriv_wrt_prev_hidden_buff, hidden_dim, hidden_dim, batch_size);

				// add hadamard(dL/dK_t, dK_t/dh_(t-1)) to existing dL/dh_(t-1) buffer
				// after all 4 passes, will be used to starting point for prior cells cell_derivs -> hidden
				my_hadamard_add(upstream_state_derivs_list[k], state_deriv_wrt_prev_hidden_buff, prev_hidden_deriv_buff, n_els);

				// special for content, need to pass back deriv to prior cell because computation depends on prior cell's content
				// will be used to set starting point for prior cell's cell_derivs -> content
				// over-writes value in prev_content_deriv_buff
				if (k == 0){
					my_hadamard(cell_derivs -> content_temp, cur_cell -> remember, prev_content_deriv_buff, n_els);
				}
			}

			// COPY BUFFERS TO NEXT PASS-THROUGH
			memcpy(cell_derivs -> hidden, prev_hidden_deriv_buff, n_els * sizeof(float));
			// need to clear to 0, otherwise next iteration will keep adding...
			memset(prev_hidden_deriv_buff, 0, n_els * sizeof(float));
			memcpy(cell_derivs -> content, prev_content_deriv_buff, n_els * sizeof(float));
		}

		// FREE THE HELPER BUFFERS...
		free(sigmoid_activ_deriv_buff);
		free(upstream_activ_deriv_buff);
		free(state_deriv_wrt_prev_hidden_buff);
		free(prev_hidden_deriv_buff);
		free(prev_content_deriv_buff);
		free(ones_hidden_dim);
		free(ones_batch_dim);


		// NOW THE GRADIENTS ARE COMPUTED IN THE GLOBAL embed_weight_derivs, bias_derivs, hidden_weight_derivs! 
		// WILL USE THESE IN OPTIMIZER TO UPDATE PARAMETERS!
		// after optimizer will need to reset the backprop buffer...

		// Accessed through: trainer -> backprop_buffer -> param_derivs -> [embed_weights|biases|hidden_weights] -> [content|remember|new_input|pass_output|classify]
		return;
}


int main(int argc, char *argv[]) {
	


	/* DEFINE DIMENSIONS */
	// (could be read from command line instead...)

	// # of classes
	int n_classes = 10000;
	int input_dim = n_classes;
	int output_dim = n_classes;
	// size of hidden state vector
	int hidden_dim = 1024;
	// size of window of prior addresses
	int seq_length = 256;

	/* INPUT FILES */
	// (could also be read from command line)
	char * ADDRESS_HISTORY_FILENAME = "../data/mind_traces/tensorflow/tflow1_addr.buffer";
	char * DELTA_MAPPINGS_FILENAME = "../data/mind_traces/delta_to_index.buffer";
	

	/* INITIALIZE MODEL */
	LSTM * model = init_lstm(input_dim, hidden_dim, output_dim, seq_length);
	
	/* INITIALIZE DATA STRUCTURES USED IN TRAINING & HYPERPARAMETERS FOR TRAINING... */
	float learning_rate = .0001;
	float mean_decay = .99;
	float var_decay = .9;
	int batch_size = 1;
	int n_epochs = 1;
	Train_LSTM * trainer = init_trainer(model, learning_rate, mean_decay, var_decay, batch_size, n_epochs);

	/* LOAD TRAINING DATA & PREPROCESS */

	// Get entire series of addresses (assumed file has consective unsigned long's packed into bytes)
	const char * address_history_filename = ADDRESS_HISTORY_FILENAME;
	unsigned long * address_history;
	int n_addresses;
	long * delta_history = read_raw_training_data(address_history_filename, &n_addresses, &address_history);

	// Build Hash Table to go from Delta -> label index by using delta_to_index buffer 
	// (assumed precomputed N_classes - 1 deltas to encode, and an extra class for None)
	// (assume the buffer is series of consective long's packed into bytes partitioned in consectutive pairs 
	// (with first byte = delta, following byte = index)
	const char * delta_mappings_filename = DELTA_MAPPINGS_FILENAME;
	// creating hash table from Delta -> index, creating 2*n_classes buckets so hopefully no collisions
	HashTable* ht = create_table(n_classes * 2);
	add_delta_to_index_mappings(ht, delta_mappings_filename);

	// delta history mapped to one-hot encoding indices
	unsigned int * encoded_deltas = (unsigned int *) calloc(n_addresses, sizeof(unsigned int));
	if (!encoded_deltas){
		fprintf(stderr, "Error: calloc");
		exit(-1);
	}

	// if the delta does not map to a one-hot encoding index (i.e. not in most frequent n_classes-1 )
	int null_ind = n_classes - 1;
	unsigned int * val;
	for (int i = 0; i < n_addresses; i++){
		val = ht_search(ht, delta_history[i]);
		if (!val){
			encoded_deltas[i] = null_ind;
		}
		else{
			encoded_deltas[i] = *val;
		}
	}

	// now we have list of N encoded deltas
	// each training sample will be: X = sequence of seq_length encoded deltas, Y = correct next encoded delta
	// makes sense to batch non-overlapping sequences to get better diversity of information intra-batch (for now using batch_size=1)

	// one hot input encoding should actually be vector of length n_classes
	// however, because it only has one value of 1 (say index i) and the rest zeroes we know that when mutliplied by weight matrix (i.e. Wx)
	// the result is just the i'th column of W. So we don't need to waste memory and can fore-go store whole one-hot encoding vector
	// also because we are always passing in seq_length consective deltas, we only need to know the first address of these

	// thus batch input X is actually [1, batch_size] and not [[n_classes, batch_size], seq_length]
	// we can do the same with mini-batch output Y where it is [1, batch_size] and not [n_classes, batch_size]


	// to build batch we can select random places in time series to predict: r = rand(seq_length, N)
	// then we get input as encoded_deltas[r - seq_length] and output as encoded_deltas[r]
	// we do this batch_size # of times

	// allocate space for one minibatch, but then overwrite contents each iteration
	mini_batch = init_general_batch(trainer, encoded_deltas, n_addresses);


	/* TRAIN MODEL! */

	for (int i = 0; i < trainer -> n_epochs; i++){
		double batches_per_epoch = ceil(((float) (N - seq_length)) / batch_size);
		for (int b = 0; b < batches_per_epoch; b++){
			// generate random batch with N = batch_size different starting points for sequences of length seq_length
			populate_batch(trainer, mini_batch);
			
			// perform forwards pass based on current model params applied to mini_batch training data
			forward_pass(trainer, mini_batch);
			
			// now look at error 
			// record loss for mini-batch

			// backpropogate the loss
			backwards_pass(trainer, mini_batch); 
			
			// apply optmizer function to change weights (using atom optmizer)
			
			
		}
		// record loss for epoch
	}

	/* SAVE MODEL! */

	// Save model params for inference (using lstm_inference.c)



	/* FREE MEMORY! */


	return 0;
}


