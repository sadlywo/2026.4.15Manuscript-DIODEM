#ifndef STREAMING_TCN_WEIGHTS_H
#define STREAMING_TCN_WEIGHTS_H

#include <stdint.h>

#define STCN_INPUT_DIM 6U
#define STCN_HIDDEN_DIM 64U
#define STCN_OUTPUT_DIM 6U
#define STCN_LATENT_DIM 8U
#define STCN_NUM_BLOCKS 4U
#define STCN_KERNEL_SIZE 3U
#define STCN_HISTORY_MAX 16U

extern const float g_stcn_input_projection_weight[STCN_HIDDEN_DIM][STCN_INPUT_DIM];
extern const float g_stcn_input_projection_bias[STCN_HIDDEN_DIM];
extern const float g_stcn_block_conv1_weight[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM][STCN_HIDDEN_DIM][STCN_KERNEL_SIZE];
extern const float g_stcn_block_conv1_bias[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM];
extern const float g_stcn_block_conv2_weight[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM][STCN_HIDDEN_DIM][STCN_KERNEL_SIZE];
extern const float g_stcn_block_conv2_bias[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM];
extern const float g_stcn_latent_weight[STCN_LATENT_DIM][STCN_HIDDEN_DIM];
extern const float g_stcn_latent_bias[STCN_LATENT_DIM];
extern const float g_stcn_gate_weight[STCN_HIDDEN_DIM][STCN_LATENT_DIM];
extern const float g_stcn_gate_bias[STCN_HIDDEN_DIM];
extern const float g_stcn_shift_weight[STCN_HIDDEN_DIM][STCN_LATENT_DIM];
extern const float g_stcn_shift_bias[STCN_HIDDEN_DIM];
extern const float g_stcn_residual_weight[STCN_OUTPUT_DIM][STCN_HIDDEN_DIM];
extern const float g_stcn_residual_bias[STCN_OUTPUT_DIM];

#endif
