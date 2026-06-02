#include "ai_inference.h"

#include <string.h>

#include "stm32h7xx_hal.h"
#include "streaming_tcn.h"

static StreamingTcnState g_tcn_state;
static bool g_tcn_initialized = false;

static void dwt_cycle_counter_init(void) {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0U;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint32_t cycles_to_us(uint32_t cycles) {
  const uint32_t cycles_per_us = SystemCoreClock / 1000000U;
  if (cycles_per_us == 0U) {
    return 0U;
  }
  return cycles / cycles_per_us;
}

bool DiodeM_AI_Init(void) {
  if (g_tcn_initialized) {
    return true;
  }

  dwt_cycle_counter_init();
  StreamingTcn_Init(&g_tcn_state);
  g_tcn_initialized = true;
  return true;
}

bool DiodeM_AI_RunStep(const float normalized_sample[DIODEM_AI_CHANNELS],
                       DiodeMAiOutput *output) {
  if (normalized_sample == 0 || output == 0) {
    return false;
  }

  const uint32_t start_cycles = DWT->CYCCNT;
  const bool ok = StreamingTcn_RunStep(&g_tcn_state, normalized_sample, output->values);
  const uint32_t elapsed_cycles = DWT->CYCCNT - start_cycles;
  if (!ok) {
    return false;
  }

  output->inference_us = cycles_to_us(elapsed_cycles);
  return true;
}

bool DiodeM_AI_RunWindow(const float normalized_window[DIODEM_AI_WINDOW_SIZE][DIODEM_AI_CHANNELS],
                         DiodeMAiOutput *output) {
  if (normalized_window == 0 || output == 0) {
    return false;
  }

  StreamingTcn_Init(&g_tcn_state);
  for (uint32_t step = 0U; step < DIODEM_AI_WINDOW_SIZE; ++step) {
    if (!DiodeM_AI_RunStep(normalized_window[step], output)) {
      return false;
    }
  }
  return true;
}
