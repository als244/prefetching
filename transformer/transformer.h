#include <stddef.h>

typedef struct {
	int input_features;
	int seq_length;
	int embed;
	int query;
	int key;
	int value;
	int ff_hidden;
	int heads;
	int n_encoders;
} Dims;

typedef struct {
	float * queries;
	float * keys;
	float * values;
} Head_Weights;

typedef struct {
	float * queries;
	float * keys;
	float * values;
	float * filter;
	float * filter_value;
} Head_Computation;

typedef struct {
	Head_Weights * weights;
	Head_Computation * computations; 
} Head;

typedef struct {
	float * heads_project;
	float * ff_0;
	float * ff_1;
	float * ff_0_bias;
	float * ff_1_bias;
} Encoder_Weights;

typedef struct {
	float * attention;
	float * attention_residual;
	float * attention_norm;
	float * ff_0;
	float * ff_1;
	float * ff_residual;
	float * encoder_out;
} Encoder_Computation;

typedef struct {
	float * input;
	Head * heads;
	Encoder_Weights * encoder_weights; 
} Encoder;


typedef struct {
	float * input_seq;
	float * embed;
	float * positional;
} Input;

typedef struct {
	float * linear_output;
	float * softmax_result;
} Output;

typedef struct {
	Input * input;
	Encoder ** encoders;
	Output * output;
} Transformer;


typedef struct {
	Transformer * model;
	Transofmrer * model_derivs;
	Transformer_Hyperparams * training_params;
} Train_Transformer;


typedef struct {
	// Number of Times to Repeat the Training Data
	int n_epochs;
	// Number of Independent Sequences to Pass to Forward Pass
	int batch_size;
	// Adam Params
	float learning_rate;
	float mean_decay;
	float var_decay;
	// Masked Language Model Training Params
	float mask_rate;
	float correct_replace_rate;
	float no_replace_rate;
	float incorrect_replace_rate;
	// Store the Training Loss Per Epoch
	float * loss;
} Transformer_Hyperparams;
