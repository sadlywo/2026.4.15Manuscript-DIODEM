#ifndef IMU_STREAM_H
#define IMU_STREAM_H

#include <stdbool.h>
#include <stdint.h>

#include "ai_inference.h"

typedef enum {
  DIODEM_STREAM_WARMUP = 0,
  DIODEM_STREAM_OK = 1,
  DIODEM_STREAM_ERROR = 2
} DiodeMStreamStatus;

typedef struct {
  float ring[DIODEM_AI_WINDOW_SIZE][DIODEM_AI_CHANNELS];
  uint32_t write_index;
  uint32_t samples_seen;
} DiodeMImuStream;

void DiodeM_Stream_Init(DiodeMImuStream *stream);
DiodeMStreamStatus DiodeM_Stream_Push(DiodeMImuStream *stream,
                                      const float raw_sample[DIODEM_AI_CHANNELS],
                                      DiodeMAiOutput *output);
void DiodeM_NormalizeWindow(const DiodeMImuStream *stream,
                            float normalized_window[DIODEM_AI_WINDOW_SIZE][DIODEM_AI_CHANNELS]);
void DiodeM_DenormalizeOutput(float output[DIODEM_AI_CHANNELS]);

#endif

