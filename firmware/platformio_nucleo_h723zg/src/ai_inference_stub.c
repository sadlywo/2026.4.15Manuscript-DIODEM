#include "ai_inference.h"

bool DiodeM_AI_Init(void) {
  return true;
}

bool DiodeM_AI_RunWindow(const float normalized_window[DIODEM_AI_WINDOW_SIZE][DIODEM_AI_CHANNELS],
                         DiodeMAiOutput *output) {
  if (normalized_window == 0 || output == 0) {
    return false;
  }

  for (unsigned int c = 0U; c < DIODEM_AI_CHANNELS; ++c) {
    output->values[c] = normalized_window[DIODEM_AI_WINDOW_SIZE - 1U][c];
  }
  output->inference_us = 0U;
  return true;
}

