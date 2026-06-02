#ifndef STREAMING_TCN_H
#define STREAMING_TCN_H

#include <stdbool.h>

#include "streaming_tcn_weights.h"

typedef struct {
  float conv1_history[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM][STCN_HISTORY_MAX];
  float conv2_history[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM][STCN_HISTORY_MAX];
} StreamingTcnState;

void StreamingTcn_Init(StreamingTcnState *state);
bool StreamingTcn_RunStep(StreamingTcnState *state,
                          const float normalized_input[STCN_INPUT_DIM],
                          float normalized_output[STCN_OUTPUT_DIM]);

#endif
