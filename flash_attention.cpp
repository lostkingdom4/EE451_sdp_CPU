#include <iostream>
#include <cmath>
#include <limits>
#include <omp.h>
#include <time.h>

using namespace std;

void flash_attention(float**** query, float**** key, float**** value, float**** output, 
                     int batch_size, int num_heads, int L, int D, int chunk_size = 64) {
    float scale_factor = 1.0 / sqrt(D);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            // Process queries in chunks
            for (int q_start = 0; q_start < L; q_start += chunk_size) {
                int q_end = min(q_start + chunk_size, L);

                // Allocate local buffers for the chunk
                float local_output[chunk_size][D] = {0};
                float local_max[chunk_size];
                float local_sum[chunk_size];

                // Initialize max and sum buffers
                for (int qi = 0; qi < q_end - q_start; ++qi) {
                    local_max[qi] = -numeric_limits<float>::infinity();
                    local_sum[qi] = 0.0;
                }

                // Process keys in chunks
                for (int k_start = 0; k_start < L; k_start += chunk_size) {
                    int k_end = min(k_start + chunk_size, L);

                    // Attention and softmax calculation for the chunk
                    for (int qi = q_start; qi < q_end; ++qi) {
                        for (int ki = k_start; ki < k_end; ++ki) {
                            float dot_product = 0.0;
                            for (int d = 0; d < D; ++d) {
                                dot_product += query[b][h][qi][d] * key[b][h][ki][d];
                            }
                            float scaled_dot = dot_product * scale_factor;

                            // Update max value for softmax stability
                            int local_idx = qi - q_start;
                            local_max[local_idx] = max(local_max[local_idx], scaled_dot);

                            // Temporarily store exponential score
                            local_sum[local_idx] += exp(scaled_dot);
                        }
                    }
                }

                // Normalize scores and compute output
                for (int qi = q_start; qi < q_end; ++qi) {
                    int local_idx = qi - q_start;
                    float normalization = local_sum[local_idx];
                    for (int ki = 0; ki < L; ++ki) {
                        float dot_product = 0.0;
                        for (int d = 0; d < D; ++d) {
                            dot_product += query[b][h][qi][d] * key[b][h][ki][d];
                        }
                        float scaled_dot = exp(dot_product * scale_factor - local_max[local_idx]) / normalization;

                        // Weighted sum for output
                        for (int d = 0; d < D; ++d) {
                            local_output[local_idx][d] += scaled_dot * value[b][h][ki][d];
                        }
                    }
                }

                // Write local output back to the global output
                for (int qi = q_start; qi < q_end; ++qi) {
                    for (int d = 0; d < D; ++d) {
                        output[b][h][qi][d] = local_output[qi - q_start][d];
                    }
                }
            }
        }
    }
}

int main() {
    int batch_size = 1;
    int num_heads = 12;
    int L = 256; // Sequence length
    int D = 1024;  // Embedding dimension

    // omp_set_num_threads(4);


    // Allocate memory
    float**** query = new float***[batch_size];
    float**** key = new float***[batch_size];
    float**** value = new float***[batch_size];
    float**** output = new float***[batch_size];

    for (int b = 0; b < batch_size; ++b) {
        query[b] = new float**[num_heads];
        key[b] = new float**[num_heads];
        value[b] = new float**[num_heads];
        output[b] = new float**[num_heads];
        for (int h = 0; h < num_heads; ++h) {
            query[b][h] = new float*[L];
            key[b][h] = new float*[L];
            value[b][h] = new float*[L];
            output[b][h] = new float*[L];
            for (int i = 0; i < L; ++i) {
                query[b][h][i] = new float[D];
                key[b][h][i] = new float[D];
                value[b][h][i] = new float[D];
                output[b][h][i] = new float[D];
                for (int j = 0; j < D; ++j) {
                    query[b][h][i][j] = static_cast<float>(rand()) / RAND_MAX;
                    key[b][h][i][j] = static_cast<float>(rand()) / RAND_MAX;
                    value[b][h][i][j] = static_cast<float>(rand()) / RAND_MAX;
                }
            }
        }
    }

    // Run Flash Attention
    struct timespec start, stop;
    double time;
    if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
        perror("clock gettime");
    }
    flash_attention(query, key, value, output, batch_size, num_heads, L, D);
    if (clock_gettime(CLOCK_REALTIME, &stop) == -1) {
        perror("clock gettime");
    }
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    cout << "Time taken for Flash Attention: " << time << " seconds" << endl;

    // Free memory
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < L; ++i) {
                delete[] query[b][h][i];
                delete[] key[b][h][i];
                delete[] value[b][h][i];
                delete[] output[b][h][i];
            }
            delete[] query[b][h];
            delete[] key[b][h];
            delete[] value[b][h];
            delete[] output[b][h];
        }
        delete[] query[b];
        delete[] key[b];
        delete[] value[b];
        delete[] output[b];
    }
    delete[] query;
    delete[] key;
    delete[] value;
    delete[] output;

    return 0;
}