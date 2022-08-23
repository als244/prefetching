#include <stddef.h>

typedef struct {
	// number of tokens in sequence
	int seq_length;
	// will assume one-hot encoded input feature length and output dim length
	// including the "mask" special token
	int classes;
	// dimension to convert input_features to internal model dim
	int embed;
	// assume dimension query=key=value=v
	int v;
	// the hidden dimension between the 2 feed forward layers
	int ff_hidden;
	// number of heads in each encoder layer
	int heads;
	// number of encoders
	int n_encoders;
} Dims;

typedef struct {
	// all (v, embed)
	float * queries;
	float * keys;
	float * values;
} Head_Weights;

typedef struct {
	// (v, seq_length)
	float * queries;
	float * keys;
	float * values;
	// (seq_length, seq_length)
	float * filter;
	// (v, seq_length)
	float * filter_value;
} Head_Computation;

typedef struct {
	Head_Weights * weights;
	Head_Computation * computations; 
} Head;

typedef struct {
	// (embed, v * heads)
	float * heads_projection;
	// (embed, ff_hidden)
	float * ff_0;
	float * ff_0_bias;
	// (ff_hidden, embed)
	float * ff_1;
	float * ff_1_bias;
} Encoder_Weights;

typedef struct {
	// ALL [(embed, seq_length), batch_size]
	float * attention;
	float * attention_residual;
	float * attention_norm;
	float * ff_0;
	float * ff_1;
	float * ff_residual;
	float * encoder_out;
} Encoder_Computation;

typedef struct {
	// will originally be from the embedded input batch, but then will be from previous layers
	// [(embed, seq_length), batch_size]
	float * input;
	// will be "heads" number of heads
	Head ** heads;
	// the weights and computations for rest of single encoder block
	Encoder_Weights * encoder_weights; 
	Encoder_Computation * encoder_computations;
} Encoder;

typedef struct {
	// ALL [(classes, seq_length), batch_size]
	// for now will treat each input sequence as different array
	// will assume that the input sequence has masked tokens
	float * input_seq;
	// if the input token is masked, will want to predict and has value of 1
	bool * is_mask;
	// the output sequence contains the mask's replacement
	float * output_seq;
} Batch;

typedef struct {
	Batch * input_batch;
	// (embed, classes)
	float * embed_weights;
	// (embed, seq_length) => will use sinusoidal family functions
	float * positional_encoding;
} Input;

typedef struct {
	// both [(classes, # of masked tokens in seq), batch_size]
	float * linear_output;
	float * softmax_result;
} Output;

typedef struct {
	Input * input;
	// will be "n_encoders" number of encoders
	Encoder ** encoders;
	Output * output;
} Transformer;


typedef struct {
	// Number of Times to Repeat the Training Data
	int n_epochs;
	// Number of Independent Sequences to Pass to Forward Pass
	int batch_size;
	// Forward pass
	Transformer * model;
	// Backprop values
	Transformer * model_derivs;
	// Optimizer History (only memory for weights will be allocated)
	Transformer * prev_means;
	Transformer * prev_vars;
	// Hyperparams for updating values
	Learning_Hyperparams * learning_hyperparams;
	// Store the Training Loss Per Epoch
	float * loss;
} Train_Transformer;


typedef struct {
	// Adam Params
	float learning_rate;
	float mean_decay;
	float var_decay;
	// Masked Language Model Training Params
	float mask_rate;
	float correct_replace_rate;
	float no_replace_rate;
	float incorrect_replace_rate;
} Learning_Hyperparams;

