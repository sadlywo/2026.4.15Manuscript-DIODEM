#ifndef AI_INFERENCE_H
#define AI_INFERENCE_H

#include <stdbool.h>
#include <stdint.h>

#define DIODEM_AI_WINDOW_SIZE 64U
#define DIODEM_AI_CHANNELS 6U

typedef struct {
  float values[DIODEM_AI_CHANNELS];
  uint32_t inference_us;
} DiodeMAiOutput;

bool DiodeM_AI_Init(void);
bool DiodeM_AI_RunWindow(const float normalized_window[DIODEM_AI_WINDOW_SIZE][DIODEM_AI_CHANNELS],
                         DiodeMAiOutput *output);

#endif

