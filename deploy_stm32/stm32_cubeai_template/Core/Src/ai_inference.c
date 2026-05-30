#include "ai_inference.h"

#include <string.h>

/*
 * STM32Cube.AI integration template.
 *
 * After generating code with X-CUBE-AI, include the generated network headers
 * and replace the `network_*` symbols below if Cube.AI uses a different model
 * name.
 */
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"

static ai_handle g_network = AI_HANDLE_NULL;
static ai_u8 g_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
static ai_float g_input[DIODEM_AI_WINDOW_SIZE * DIODEM_AI_CHANNELS];
static ai_float g_output[DIODEM_AI_CHANNELS];

static uint32_t DiodeM_ReadCycles(void) {
#if defined(DWT) && defined(CoreDebug_DEMCR_TRCENA_Msk)
  return DWT->CYCCNT;
#else
  return 0U;
#endif
}

bool DiodeM_AI_Init(void) {
  ai_error error;
  const ai_handle acts[] = {g_activations};

  error = ai_network_create(&g_network, AI_NETWORK_DATA_CONFIG);
  if (error.type != AI_ERROR_NONE) {
    return false;
  }

  ai_network_params params = {
      AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
      AI_NETWORK_DATA_ACTIVATIONS(acts)};

  if (!ai_network_init(g_network, &params)) {
    return false;
  }

#if defined(DWT) && defined(CoreDebug_DEMCR_TRCENA_Msk)
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
#endif
  return true;
}

bool DiodeM_AI_RunWindow(const float normalized_window[DIODEM_AI_WINDOW_SIZE][DIODEM_AI_CHANNELS],
                         DiodeMAiOutput *output) {
  if (g_network == AI_HANDLE_NULL || normalized_window == NULL || output == NULL) {
    return false;
  }

  uint32_t flat = 0U;
  for (uint32_t t = 0U; t < DIODEM_AI_WINDOW_SIZE; ++t) {
    for (uint32_t c = 0U; c < DIODEM_AI_CHANNELS; ++c) {
      g_input[flat++] = normalized_window[t][c];
    }
  }

  ai_buffer ai_input[AI_NETWORK_IN_NUM] = AI_NETWORK_IN;
  ai_buffer ai_output[AI_NETWORK_OUT_NUM] = AI_NETWORK_OUT;
  ai_input[0].data = AI_HANDLE_PTR(g_input);
  ai_output[0].data = AI_HANDLE_PTR(g_output);

  uint32_t start_cycles = DiodeM_ReadCycles();
  ai_i32 batch = ai_network_run(g_network, ai_input, ai_output);
  uint32_t elapsed_cycles = DiodeM_ReadCycles() - start_cycles;
  if (batch != 1) {
    return false;
  }

  memcpy(output->values, g_output, sizeof(g_output));
#if defined(SystemCoreClock)
  output->inference_us = (uint32_t)(((uint64_t)elapsed_cycles * 1000000ULL) / SystemCoreClock);
#else
  output->inference_us = elapsed_cycles;
#endif
  return true;
}

