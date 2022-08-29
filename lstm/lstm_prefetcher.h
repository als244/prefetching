#include <stddef.h>

#include "hash_table.h"

/* STRUCTURES */

typedef struct {
	int input;
	int hidden;
	int output;
	int seq_length;
} Dims;

typedef struct {
	float * content;
	float * remember;
	float * new_input;
	float * pass_output;
	float * classify;
} Embed_Weights;

typedef struct {
	float * content;
	float * remember;
	float * new_input;
	float * pass_output;
	float * classify;
} Biases;

typedef struct {
	float * content;
	float * remember;
	float * new_input;
	float * pass_output;
} Hidden_Weights;

typedef struct {
	Embed_Weights* embed_weights;
	Biases* biases;
	Hidden_Weights* hidden_weights;
} Params;

typedef struct {
	Dims dims;
	Params* params;
} LSTM;

typedef struct {
	float * content_temp;
	float * content;
	float * remember;
	float * new_input;
	float * pass_output;
	float * hidden;
} LSTM_Cell;

typedef struct {
	LSTM_Cell ** cells;
	float * linear_output;
	float * label_distribution;
} Forward_Buffer;

// The previous hidden state deriv will be the starting point for each timestep's local backprop. (dL/dh_(t-1)) 
// The previous content state needs to be added to the local computation of the content step when going backwards (dL/dc_(t-1))
typedef struct {
	float * output_layer_deriv;
	Params * param_derivs;
	Params * prev_means;
	Params * prev_vars;
	LSTM_Cell * cell_derivs;
} Backprop_Buffer;

typedef struct {
	LSTM * model;
	Forward_Buffer * forward_buffer;
	Backprop_Buffer * backprop_buffer;
	float learning_rate;
	float mean_decay;
	float var_decay;
	int batch_size;
	int n_epochs;
	float loss;
} Train_LSTM;

typedef struct{
	unsigned int * training_ind_seq_start;
	unsigned int * current_input_token_ids;
	unsigned int * correct_label_encoded;
	unsigned int ** training_data;
	int training_data_length;
} Batch;


/* FUNCTION DECLARATIONS */
float sample_gaussian(float mean, float var);
Params * init_model_parameters(Dims model_dims, bool is_zero);
void init_weights(float *weights, int size, int unit_inputs, int unit_outputs);
LSTM_Cell * init_lstm_cell(Dims model_dims, int batch_size);
Forward_Buffer * init_forward_buffer(Dims model_dims, int batch_size);
Backprop_Buffer * init_backprop_buffer(Dims model_dims, int batch_size);
LSTM * init_lstm(int input_dim, int hidden_dim, int output_dim, int seq_length);
Train_LSTM * init_trainer(LSTM * model, float learning_rate, int batch_size);
long * read_raw_training_data(const char * filename, int * n_addresses, unsigned long ** address_history);
void add_delta_to_index_mappings(HashTable * ht, const char * filename);
Batch * init_general_batch(Train_LSTM * trainer, unsigned int * training_data, int training_data_length);
void populate_batch(Train_LSTM * trainer, Batch * mini_batch);
void forward_pass(Train_LSTM * trainer, Batch * mini_batch);
