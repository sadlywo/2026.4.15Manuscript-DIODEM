#include "imu_stream.h"

#include <stddef.h>

static const float kInputMean[DIODEM_AI_CHANNELS] = {
    -0.0121626081f, -1.0057030916f, 3.5020763874f,
    -0.0098095555f, -0.0054507395f, 0.0020694262f};

static const float kInputStd[DIODEM_AI_CHANNELS] = {
    5.9565930367f, 5.9830369949f, 5.0655570030f,
    1.9069567919f, 1.2362844944f, 0.9681030512f};

static const float kTargetMean[DIODEM_AI_CHANNELS] = {
    -0.0077581760f, -1.2969622612f, 4.3221783638f,
    -0.0111715924f, -0.0074166609f, 0.0015425781f};

static const float kTargetStd[DIODEM_AI_CHANNELS] = {
    5.8246135712f, 5.3729209900f, 4.6909785271f,
    1.7278544903f, 1.1676394939f, 0.9388045669f};

void DiodeM_Stream_Init(DiodeMImuStream *stream) {
  if (stream == NULL) {
    return;
  }
  stream->write_index = 0U;
  stream->samples_seen = 0U;
  for (uint32_t t = 0U; t < DIODEM_AI_WINDOW_SIZE; ++t) {
    for (uint32_t c = 0U; c < DIODEM_AI_CHANNELS; ++c) {
      stream->ring[t][c] = 0.0f;
    }
  }
}

void DiodeM_NormalizeWindow(const DiodeMImuStream *stream,
                            float normalized_window[DIODEM_AI_WINDOW_SIZE][DIODEM_AI_CHANNELS]) {
  uint32_t oldest = stream->write_index;
  for (uint32_t t = 0U; t < DIODEM_AI_WINDOW_SIZE; ++t) {
    uint32_t source_index = (oldest + t) % DIODEM_AI_WINDOW_SIZE;
    for (uint32_t c = 0U; c < DIODEM_AI_CHANNELS; ++c) {
      normalized_window[t][c] = (stream->ring[source_index][c] - kInputMean[c]) / kInputStd[c];
    }
  }
}

void DiodeM_DenormalizeOutput(float output[DIODEM_AI_CHANNELS]) {
  for (uint32_t c = 0U; c < DIODEM_AI_CHANNELS; ++c) {
    output[c] = output[c] * kTargetStd[c] + kTargetMean[c];
  }
}

DiodeMStreamStatus DiodeM_Stream_Push(DiodeMImuStream *stream,
                                      const float raw_sample[DIODEM_AI_CHANNELS],
                                      DiodeMAiOutput *output) {
  float normalized_sample[DIODEM_AI_CHANNELS];

  if (stream == NULL || raw_sample == NULL || output == NULL) {
    return DIODEM_STREAM_ERROR;
  }

  for (uint32_t c = 0U; c < DIODEM_AI_CHANNELS; ++c) {
    stream->ring[stream->write_index][c] = raw_sample[c];
    normalized_sample[c] = (raw_sample[c] - kInputMean[c]) / kInputStd[c];
  }
  stream->write_index = (stream->write_index + 1U) % DIODEM_AI_WINDOW_SIZE;
  stream->samples_seen += 1U;

  if (!DiodeM_AI_RunStep(normalized_sample, output)) {
    return DIODEM_STREAM_ERROR;
  }

  if (stream->samples_seen < DIODEM_AI_WINDOW_SIZE) {
    return DIODEM_STREAM_WARMUP;
  }

  DiodeM_DenormalizeOutput(output->values);
  return DIODEM_STREAM_OK;
}
