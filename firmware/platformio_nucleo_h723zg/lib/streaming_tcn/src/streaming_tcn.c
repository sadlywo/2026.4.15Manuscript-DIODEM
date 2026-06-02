#include "streaming_tcn.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static float relu(float value) {
  return value > 0.0f ? value : 0.0f;
}

static float sigmoidf_stable(float value) {
  return 1.0f / (1.0f + expf(-value));
}

static void linear_64x6(const float weight[STCN_HIDDEN_DIM][STCN_INPUT_DIM],
                        const float bias[STCN_HIDDEN_DIM],
                        const float input[STCN_INPUT_DIM],
                        float output[STCN_HIDDEN_DIM]) {
  for (uint32_t out_ch = 0U; out_ch < STCN_HIDDEN_DIM; ++out_ch) {
    float acc = bias[out_ch];
    for (uint32_t in_ch = 0U; in_ch < STCN_INPUT_DIM; ++in_ch) {
      acc += weight[out_ch][in_ch] * input[in_ch];
    }
    output[out_ch] = acc;
  }
}

static void linear_8x64(const float weight[STCN_LATENT_DIM][STCN_HIDDEN_DIM],
                        const float bias[STCN_LATENT_DIM],
                        const float input[STCN_HIDDEN_DIM],
                        float output[STCN_LATENT_DIM]) {
  for (uint32_t out_ch = 0U; out_ch < STCN_LATENT_DIM; ++out_ch) {
    float acc = bias[out_ch];
    for (uint32_t in_ch = 0U; in_ch < STCN_HIDDEN_DIM; ++in_ch) {
      acc += weight[out_ch][in_ch] * input[in_ch];
    }
    output[out_ch] = acc;
  }
}

static void linear_64x8(const float weight[STCN_HIDDEN_DIM][STCN_LATENT_DIM],
                        const float bias[STCN_HIDDEN_DIM],
                        const float input[STCN_LATENT_DIM],
                        float output[STCN_HIDDEN_DIM]) {
  for (uint32_t out_ch = 0U; out_ch < STCN_HIDDEN_DIM; ++out_ch) {
    float acc = bias[out_ch];
    for (uint32_t in_ch = 0U; in_ch < STCN_LATENT_DIM; ++in_ch) {
      acc += weight[out_ch][in_ch] * input[in_ch];
    }
    output[out_ch] = acc;
  }
}

static void linear_6x64(const float weight[STCN_OUTPUT_DIM][STCN_HIDDEN_DIM],
                        const float bias[STCN_OUTPUT_DIM],
                        const float input[STCN_HIDDEN_DIM],
                        float output[STCN_OUTPUT_DIM]) {
  for (uint32_t out_ch = 0U; out_ch < STCN_OUTPUT_DIM; ++out_ch) {
    float acc = bias[out_ch];
    for (uint32_t in_ch = 0U; in_ch < STCN_HIDDEN_DIM; ++in_ch) {
      acc += weight[out_ch][in_ch] * input[in_ch];
    }
    output[out_ch] = acc;
  }
}

static void update_history(float history[STCN_HIDDEN_DIM][STCN_HISTORY_MAX],
                           const float current[STCN_HIDDEN_DIM],
                           uint32_t history_len) {
  for (uint32_t channel = 0U; channel < STCN_HIDDEN_DIM; ++channel) {
    for (uint32_t index = 0U; index + 1U < history_len; ++index) {
      history[channel][index] = history[channel][index + 1U];
    }
    history[channel][history_len - 1U] = current[channel];
  }
}

static void conv_step(
    const float weight[STCN_HIDDEN_DIM][STCN_HIDDEN_DIM][STCN_KERNEL_SIZE],
    const float bias[STCN_HIDDEN_DIM],
    float history[STCN_HIDDEN_DIM][STCN_HISTORY_MAX],
    const float current[STCN_HIDDEN_DIM],
    uint32_t dilation,
    float output[STCN_HIDDEN_DIM]) {
  for (uint32_t out_ch = 0U; out_ch < STCN_HIDDEN_DIM; ++out_ch) {
    float acc = bias[out_ch];
    for (uint32_t in_ch = 0U; in_ch < STCN_HIDDEN_DIM; ++in_ch) {
      acc += weight[out_ch][in_ch][0] * history[in_ch][0];
      acc += weight[out_ch][in_ch][1] * history[in_ch][dilation];
      acc += weight[out_ch][in_ch][2] * current[in_ch];
    }
    output[out_ch] = acc;
  }
}

void StreamingTcn_Init(StreamingTcnState *state) {
  if (state == NULL) {
    return;
  }
  memset(state, 0, sizeof(*state));
}

bool StreamingTcn_RunStep(StreamingTcnState *state,
                          const float normalized_input[STCN_INPUT_DIM],
                          float normalized_output[STCN_OUTPUT_DIM]) {
  float features[STCN_HIDDEN_DIM];
  float residual_features[STCN_HIDDEN_DIM];
  float conv1[STCN_HIDDEN_DIM];
  float conv2[STCN_HIDDEN_DIM];
  float latent[STCN_LATENT_DIM];
  float gate[STCN_HIDDEN_DIM];
  float shift[STCN_HIDDEN_DIM];
  float conditioned[STCN_HIDDEN_DIM];
  float residual[STCN_OUTPUT_DIM];

  if (state == NULL || normalized_input == NULL || normalized_output == NULL) {
    return false;
  }

  linear_64x6(g_stcn_input_projection_weight, g_stcn_input_projection_bias, normalized_input, features);

  for (uint32_t block = 0U; block < STCN_NUM_BLOCKS; ++block) {
    const uint32_t dilation = 1U << block;
    const uint32_t history_len = 2U * dilation;
    memcpy(residual_features, features, sizeof(residual_features));

    conv_step(g_stcn_block_conv1_weight[block],
              g_stcn_block_conv1_bias[block],
              state->conv1_history[block],
              features,
              dilation,
              conv1);
    for (uint32_t channel = 0U; channel < STCN_HIDDEN_DIM; ++channel) {
      conv1[channel] = relu(conv1[channel]);
    }
    update_history(state->conv1_history[block], features, history_len);

    conv_step(g_stcn_block_conv2_weight[block],
              g_stcn_block_conv2_bias[block],
              state->conv2_history[block],
              conv1,
              dilation,
              conv2);
    for (uint32_t channel = 0U; channel < STCN_HIDDEN_DIM; ++channel) {
      conv2[channel] = relu(conv2[channel]);
    }
    update_history(state->conv2_history[block], conv1, history_len);

    for (uint32_t channel = 0U; channel < STCN_HIDDEN_DIM; ++channel) {
      features[channel] = conv2[channel] + residual_features[channel];
    }
  }

  linear_8x64(g_stcn_latent_weight, g_stcn_latent_bias, features, latent);
  linear_64x8(g_stcn_gate_weight, g_stcn_gate_bias, latent, gate);
  linear_64x8(g_stcn_shift_weight, g_stcn_shift_bias, latent, shift);
  for (uint32_t channel = 0U; channel < STCN_HIDDEN_DIM; ++channel) {
    gate[channel] = sigmoidf_stable(gate[channel]);
    conditioned[channel] = features[channel] * (1.0f + gate[channel]) + shift[channel];
  }

  linear_6x64(g_stcn_residual_weight, g_stcn_residual_bias, conditioned, residual);
  for (uint32_t channel = 0U; channel < STCN_OUTPUT_DIM; ++channel) {
    normalized_output[channel] = normalized_input[channel] + residual[channel];
  }
  return true;
}
