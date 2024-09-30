// Include necessary C++ standard libraries
#include <vector>      // For using dynamic arrays (vectors)
#include <fstream>     // For file input/output operations
#include <cmath>       // For mathematical functions like sqrt, exp
#include <random>      // For random number generation
#include <iostream>    // For input/output operations
#include <string>      // For string manipulation

// Define custom types for convenience and readability
typedef std::vector<float> tensor1d;           // 1-dimensional tensor (vector of floats)
typedef std::vector<tensor1d> tensor2d;        // 2-dimensional tensor (vector of vectors)
typedef std::vector<tensor2d> tensor3d;        // 3-dimensional tensor (vector of 2D tensors)

// Define a small constant for numerical stability
float EPS = 1e-5;  // Used to prevent division by zero in some calculations

// Define the configuration structure for the transformer model
struct Config {
    int dim;           // Dimension of the transformer (size of embeddings)
    int hidden_dim;    // Dimension of the hidden layer in feed-forward networks
    int n_layers;      // Number of transformer layers
    int n_heads;       // Number of attention heads
    int n_kv_heads;    // Number of key/value heads (can be less than query heads in multi-query attention)
    int vocab_size;    // Size of the vocabulary
    int seq_len;       // Maximum sequence length the model can handle
};

// Define the structure to hold all weights of the transformer
struct TransformerWeights {
    tensor2d token_embedding_table;  // Embeddings for each token in the vocabulary [vocab_size, dim]
    // Weights for layer normalization (root mean square normalization)
    tensor2d rms_att_weight;  // [layer, dim]
    tensor2d rms_ffn_weight;  // [layer, dim]
    // Weights for attention mechanisms
    tensor3d wq;  // Query weights [layer, dim, dim]
    tensor3d wk;  // Key weights [layer, dim, dim]
    tensor3d wv;  // Value weights [layer, dim, dim]
    tensor3d wo;  // Output weights [layer, dim, dim]
    // Weights for feed-forward networks
    tensor3d w1;  // First linear layer [layer, hidden_dim, dim]
    tensor3d w2;  // Second linear layer [layer, dim, hidden_dim]
    tensor3d w3;  // Third linear layer (for SwiGLU activation) [layer, hidden_dim, dim]
    // Final layer normalization
    tensor1d rms_final_weight;  // [dim]
    // Rotary positional embeddings
    tensor2d freq_cis_real;  // Real part [seq_len, (dim/n_heads)/2]
    tensor2d freq_cis_imag;  // Imaginary part [seq_len, (dim/n_heads)/2]
};

// Define the structure to hold the current state of the transformer during inference
struct RunState {
    // Activations at the current timestamp
    tensor1d x;         // Current input/output activations [dim]
    tensor1d xb;        // Temporary buffer for residual connections [dim]
    tensor1d xb2;       // Another temporary buffer [dim]
    tensor1d hb;        // Hidden state in feed-forward network [hidden_dim]
    tensor1d hb2;       // Another hidden state buffer [hidden_dim]
    tensor1d q;         // Query vector for attention [dim]
    tensor1d k;         // Key vector for attention [dim]
    tensor1d v;         // Value vector for attention [dim]
    tensor1d attention; // Attention scores [seq_len]
    tensor1d logits;    // Output logits [vocab_size]
    // Key-Value cache for efficient inference
    tensor3d key_cache;    // Cached keys [layer, seq_len, dim]
    tensor3d value_cache;  // Cached values [layer, seq_len, dim]
};

// Function to resize the state tensors based on the configuration
void resize_state_tensors(RunState &state, Config &config) {
    // Resize all tensors in the RunState to match the configuration
    // This ensures that we have the correct dimensions for all our buffers
    state.x.resize(config.dim);
    state.xb.resize(config.dim);
    state.xb2.resize(config.dim);
    state.hb.resize(config.hidden_dim);
    state.hb2.resize(config.hidden_dim);
    state.q.resize(config.dim);
    state.k.resize(config.dim);
    state.v.resize(config.dim);
    state.attention.resize(config.seq_len);
    state.logits.resize(config.vocab_size);
    state.key_cache.resize(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim)));
    state.value_cache.resize(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim)));
}

// Function to free memory used by state tensors
void free_state_tensors(RunState &state) {
    // Clear all vectors to free up memory
    // This is not strictly necessary in modern C++ as vectors automatically free their memory when they go out of scope
    // However, it can be useful for immediate memory reclamation in long-running programs
    state.x.clear();
    state.xb.clear();
    state.xb2.clear();
    state.hb.clear();
    state.hb2.clear();
    state.q.clear();
    state.k.clear();
    state.v.clear();
    state.attention.clear();
    state.logits.clear();
    state.key_cache.clear();
    state.value_cache.clear();
}

// Function to resize the weights tensors based on the configuration
void resize_weights_tensors(TransformerWeights &weights, Config &config) {
    // Resize all weight tensors to match the configuration
    // This ensures that we have the correct dimensions for all our model parameters
    weights.token_embedding_table.resize(config.vocab_size, tensor1d(config.dim));
    weights.rms_att_weight.resize(config.n_layers, tensor1d(config.dim));
    weights.rms_ffn_weight.resize(config.n_layers, tensor1d(config.dim));
    weights.wq.resize(config.n_layers, tensor2d(config.dim, tensor1d(config.dim)));
    weights.wk.resize(config.n_layers, tensor2d(config.dim, tensor1d(config.dim)));
    weights.wv.resize(config.n_layers, tensor2d(config.dim, tensor1d(config.dim)));
    weights.wo.resize(config.n_layers, tensor2d(config.dim, tensor1d(config.dim)));
    weights.w1.resize(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim)));
    weights.w2.resize(config.n_layers, tensor2d(config.dim, tensor1d(config.hidden_dim)));
    weights.w3.resize(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim)));
    weights.rms_final_weight.resize(config.dim);
    int head_size = config.dim / config.n_heads;
    weights.freq_cis_real.resize(config.seq_len, tensor1d(head_size / 2));
    weights.freq_cis_imag.resize(config.seq_len, tensor1d(head_size / 2));
}

// Function to free memory used by weights tensors
void free_weights_tensors(TransformerWeights &weights) {
    // Clear all vectors to free up memory
    weights.token_embedding_table.clear();
    weights.rms_att_weight.clear();
    weights.rms_ffn_weight.clear();
    weights.wq.clear();
    weights.wk.clear();
    weights.wv.clear();
    weights.wo.clear();
    weights.w1.clear();
    weights.w2.clear();
    weights.w3.clear();
    weights.rms_final_weight.clear();
    weights.freq_cis_real.clear();
    weights.freq_cis_imag.clear();
}

// Functions to initialize tensors from a checkpoint file
// These functions read binary data from a file stream and populate the tensors

void checkpoint_init_tensor(tensor1d &tensor, std::fstream &file) {
    // Read binary data directly into the tensor
    file.read((char*)(tensor.data()), tensor.size() * sizeof(float));
}

void checkpoint_init_tensor(tensor2d &tensor, std::fstream &file) {
    // Initialize each row of the 2D tensor
    for (auto &t : tensor) checkpoint_init_tensor(t, file);
}

void checkpoint_init_tensor(tensor3d &tensor, std::fstream &file) {
    // Initialize each 2D tensor in the 3D tensor
    for (auto &t : tensor) checkpoint_init_tensor(t, file);
}

// Function to initialize all weights from a checkpoint file
void checkpoint_init_weights(TransformerWeights &weights, Config &config, std::fstream &file) {
    // Read all weights from the file in the correct order
    checkpoint_init_tensor(weights.token_embedding_table, file);
    checkpoint_init_tensor(weights.rms_att_weight, file);
    checkpoint_init_tensor(weights.wq, file);
    checkpoint_init_tensor(weights.wk, file);
    checkpoint_init_tensor(weights.wv, file);
    checkpoint_init_tensor(weights.wo, file);
    checkpoint_init_tensor(weights.rms_ffn_weight, file);
    checkpoint_init_tensor(weights.w1, file);
    checkpoint_init_tensor(weights.w2, file);
    checkpoint_init_tensor(weights.w3, file);
    checkpoint_init_tensor(weights.rms_final_weight, file);
    checkpoint_init_tensor(weights.freq_cis_real, file);
    checkpoint_init_tensor(weights.freq_cis_imag, file);
}

// Utility functions for tensor operations

// Copy functions for different tensor dimensions
void copy(tensor1d &dst, tensor1d &src) {
    for (int i = 0; i < dst.size(); i++)  dst[i] = src[i];
}

void copy(tensor2d &dst, tensor2d &src) {
    for (int i = 0; i < dst.size(); i++)  copy(dst[i], src[i]);
}

void copy(tensor3d &dst, tensor3d &src) {
    for (int i = 0; i < dst.size(); i++)  copy(dst[i], src[i]);
}

// Function to accumulate (add) one tensor to another
void accum(tensor1d &lhs, tensor1d &rhs) {
    for (int i = 0; i < rhs.size(); ++i)  lhs[i] += rhs[i];
}

// Root Mean Square (RMS) Normalization function
void rmsnorm(tensor1d &output, tensor1d &input, tensor1d &weight) {
    float ss = 0.0;
    // Calculate sum of squares
    for (int i = 0; i < input.size(); i++)
        ss += input[i] * input[i];
    ss = ss / input.size() + EPS;  // Add epsilon for numerical stability
    float inv_ss = 1 / sqrt(ss);   // Inverse of root mean square
    // Apply normalization and scaling
    for (int i = 0; i < input.size(); i++)
        output[i] = input[i] * inv_ss * weight[i];
}

// Softmax function
void softmax(tensor1d &output, tensor1d &input, int max_pos = -1) {
    if (max_pos == -1)  max_pos = input.size();
    // Find maximum value for numerical stability
    float max_val = input[0];
    for (int i = 1; i < max_pos; i++)
        if (input[i] > max_val)  max_val = input[i];
    
    // Compute exp and sum
    float sum = 0;
    for (int i = 0; i < max_pos; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    // Normalize
    for (int i = 0; i < max_pos; i++)
        output[i] /= sum;
}

// Matrix multiplication function
void matmul(tensor1d &output, tensor1d &input, tensor2d &weight) {
    for (int i = 0; i < output.size(); i++) {
        output[i] = 0;
        for (int j = 0; j < input.size(); j++)
            output[i] += input[j] * weight[i][j];
    }
}

// Main transformer function that processes a single token
void transformer(int token_index, int token_position, Config &config, RunState &state, TransformerWeights &transformer_weights) {
    // Convenience variables
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int head_size = dim / config.n_heads;

    // Embed the input token
    copy(state.x, transformer_weights.token_embedding_table[token_index]);

    // Process through each layer of the transformer
    for (int layer = 0; layer < config.n_layers; ++layer) {
        // Attention mechanism
        rmsnorm(state.xb, state.x, transformer_weights.rms_att_weight[layer]);
        
        // Compute query, key, and value
        matmul(state.q, state.xb, transformer_weights.wq[layer]);
        matmul(state.k, state.xb, transformer_weights.wk[layer]);
        matmul(state.v, state.xb, transformer_weights.wv[layer]);

        // Apply rotary positional embeddings (RoPE)
        for (int head = 0; head < config.n_heads; ++head) {
            int start = head * head_size;
            for (int i = 0; i < head_size; i += 2) {
                float q0 = state.q[start + i];
                float q1 = state.q[start + i + 1];
                float k0 = state.k[start + i];
                float k1 = state.k[start + i + 1];
                float fcr = transformer_weights.freq_cis_real[token_position][i / 2];
                float fci = transformer_weights.freq_cis_imag[token_position][i / 2];
                state.q[start + i]     = q0 * fcr - q1 * fci;
                state.q[start + i + 1] = q0 * fci + q1 * fcr;
                state.k[start + i]     = k0 * fcr - k1 * fci;
                state.k[start + i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // Save key and value in cache for future use
        copy(state.key_cache[layer][token_position], state.k);
        copy(state.value_cache[layer][token_position], state.v);

        // Perform multi-query attention
        for (int head = 0; head < config.n_heads; ++head) {
            // Compute attention scores
            for (int timestep = 0; timestep < token_position; ++timestep) {
                float score = 0;
                for (int i = 0; i < head_size; ++i)
                    score += state.q[head * head_size + i] * state.key_cache[layer][timestep][head * head_size + i];
                score /= std::sqrt(head_size * 1.0);
                state.attention[timestep] = score;
            }

            // Apply softmax to get attention weights
            softmax(state.attention, state.attention, token_position+1);

            // Compute weighted sum of values
            for (int i = 0; i < head_size; ++i) {
                state.xb[head * head_size + i] = 0;
                for (int timestep = 0; timestep <= token_position; ++timestep)
                    state.xb[head * head_size + i] += state.attention[timestep] * state.value_cache[layer][timestep][head * head_size + i];
            }
        }

        // Final projection for attention output
        matmul(state.xb2, state.xb, transformer_weights.wo[layer]);

        // Residual connection
        accum(state.x, state.xb2);

        // Feed-forward network
        rmsnorm(state.xb, state.x, transformer_weights.rms_ffn_weight[layer]);

        // Compute feed-forward network: self.w2(F.silu(self.w1(x))) * self.w3(x)
        matmul(state.hb, state.xb, transformer_weights.w1[layer]);
        matmul(state.hb2, state.xb, transformer_weights.w3[layer]);

        // Apply SiLU (Sigmoid Linear Unit) activation function
        for (int i = 0; i < hidden_dim; ++i)
            state.hb[i] = state.hb[i] * (1.0 / (1.0 + std::exp(-state.hb[i])));

        // Element-wise multiplication
        for (int i = 0; i < hidden_dim; ++i)
            state.hb[i] = state.hb[i] * state.hb2[i];
        
        // Final projection for feed-forward network
        matmul(state.xb, state.hb, transformer_weights.w2[layer]);

        // Residual connection
        accum(state.x, state.xb);
    }

    // Final layer normalization
    rmsnorm(state.x, state.x, transformer_weights.rms_final_weight);

    // Compute logits by projecting to vocabulary size
    matmul(state.logits, state.x, transformer_weights.token_embedding_table);
}

// Utility function to sample from a probability distribution
int sample(tensor1d &probabilities) {
    // Set up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    float r = dis(gen);

    // Compute cumulative distribution and sample
    float cdf = 0.0;
    for (int i = 0; i < probabilities.size(); ++i) {
        cdf += probabilities[i];
        if (r < cdf)
            return i;
    }
    // In case of rounding errors, return last index
    return probabilities.size() - 1;
}

// Utility function to find the index of the maximum value
int argmax(tensor1d &values) {
    int max_i = 0;
    float max_value = values[0];
    for (int i = 1; i < values.size(); ++i)
        if (values[i] > max_value) {
            max_i = i;
            max_value = values[i];
        }
    return max_i;
}

// Main function
int main(int argc, char *argv[]) {
    // Disable output buffering for immediate console output
    std::cout.tie(NULL);

    // Parse command line arguments
    std::string checkpoint;
    float temperature = 0.9;
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <checkpoint_file> [temperature]\n";
        return 1;
    }
    checkpoint = argv[1];
    if (argc >= 3)
        temperature = std::atof(argv[2]);

    // Load model configuration and weights
    Config config;
    TransformerWeights transformer_weights;
    {
        std::fstream file(checkpoint);
        if (!file) {
            std::cout << "Unable to open the checkpoint file " << checkpoint << "\n";
            return 1;
        }
        file.read((char*)&config, sizeof(config));
        resize_weights_tensors(transformer_weights, config);
        checkpoint_init_weights(transformer_weights, config, file);
        file.close();
    }

    // Load vocabulary
    std::vector<std::string> vocab(config.vocab_size);
    {
        std::fstream file("tokenizer.bin");
        if (!file) {
            std::cout
                << "Unable to open the tokenizer file tokenizer.bin! Run \n"
                << "python tokenizer.py to convert tokenizer.model -> tokenizer.bin\n";
            return 1;
        }
        for (int i = 0; i < config.vocab_size; i++) {
            int len;
            vocab[i] = "";
            file.read((char*)&len, sizeof(int));
            for (int j = 0; j < len; ++j) {
                char c;
                file.read((char*)&c, sizeof(char));
                vocab[i].push_back(c);
            }
            vocab[i].push_back('\0');
        }
        file.close();
    }

    // Initialize run state
    RunState state;
    resize_state_tensors(state, config);

    // Generate text
    clock_t start = clock();
    int next;
    int token = 1;  // 1 = BOS (Beginning of Sentence) token in Llama-2 sentence-piece
    for (int pos = 0; pos < config.seq_len; ++pos) {
        // Forward pass through the transformer
        transformer(token, pos, config, state, transformer_weights);

        // Sample next token
        if (temperature < EPS) {
            next = argmax(state.logits);
        } else {
            for (int q = 0; q < config.vocab_size; ++q)
                state.logits[q] /= temperature;
            softmax(state.logits, state.logits);
            next = sample(state.logits);
        }
        std::cout << vocab[next];

        token = next;
    }
    std::cout << "\n";

    // Report generation speed
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("achieved tok/s: %f\n", config.seq_len / elapsed);

    // Clean up memory
    free_state_tensors(state);
    free_weights_tensors(transformer_weights);
    vocab.clear();

    return 0;
}
