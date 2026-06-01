#include "stm32h7xx_hal.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imu_stream.h"

#define UART_TX_GUARD_LIMIT 1000000U
#define USART_READY_GUARD_LIMIT 1000000U

static DiodeMImuStream g_stream;

static void MX_GPIO_Init(void);
static void MX_USART3_Polling_Init(void);
static void Error_Handler(void);
static void uart_write(const char *text);
static bool uart_write_char(char ch);
static bool uart_read_char(uint8_t *ch);
static bool uart_read_line(char *buffer, size_t buffer_size);
static bool parse_sample_csv(const char *line, float sample[DIODEM_AI_CHANNELS]);
static void print_result(uint32_t seq, DiodeMStreamStatus status, const DiodeMAiOutput *output);
static void fault_loop(const char *code);

int main(void) {
  HAL_Init();
  MX_GPIO_Init();
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);
  MX_USART3_Polling_Init();

  DiodeM_Stream_Init(&g_stream);
  if (!DiodeM_AI_Init()) {
    Error_Handler();
  }

  HAL_Delay(20U);
  uart_write("BOOT_DIAG_V3\r\n");
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);

  uint32_t seq = 0U;
  uint32_t last_heartbeat = HAL_GetTick();
  char line[160];
  float sample[DIODEM_AI_CHANNELS];
  DiodeMAiOutput output;

  while (1) {
    if (uart_read_line(line, sizeof(line))) {
      if (parse_sample_csv(line, sample)) {
        DiodeMStreamStatus status = DiodeM_Stream_Push(&g_stream, sample, &output);
        print_result(seq++, status, &output);
      } else {
        uart_write("ERR,bad_csv\r\n");
      }
    }

    if ((HAL_GetTick() - last_heartbeat) >= 1000U) {
      last_heartbeat = HAL_GetTick();
      HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_0);
      uart_write("HB3\r\n");
    }
  }
}

static void uart_write(const char *text) {
  if (text == NULL) {
    return;
  }
  while (*text != '\0') {
    if (!uart_write_char(*text++)) {
      return;
    }
  }
}

static bool uart_write_char(char ch) {
  for (uint32_t guard = 0U; guard < UART_TX_GUARD_LIMIT; ++guard) {
    if ((USART3->ISR & USART_ISR_TXE_TXFNF) != 0U) {
      USART3->TDR = (uint8_t)ch;
      return true;
    }
  }
  return false;
}

static bool uart_read_char(uint8_t *ch) {
  if (ch == NULL) {
    return false;
  }
  if ((USART3->ISR & USART_ISR_RXNE_RXFNE) == 0U) {
    return false;
  }
  *ch = (uint8_t)(USART3->RDR & 0xFFU);
  return true;
}

static bool uart_read_line(char *buffer, size_t buffer_size) {
  static size_t index = 0U;
  uint8_t ch = 0U;
  if (buffer == NULL || buffer_size < 2U) {
    return false;
  }

  while (uart_read_char(&ch)) {
    if (ch == '\r') {
      continue;
    }
    if (ch == '\n') {
      buffer[index] = '\0';
      index = 0U;
      return true;
    }
    if (index < buffer_size - 1U) {
      buffer[index++] = (char)ch;
    } else {
      index = 0U;
      buffer[0] = '\0';
      return false;
    }
  }
  return false;
}

static bool parse_sample_csv(const char *line, float sample[DIODEM_AI_CHANNELS]) {
  char local[160];
  char *cursor = local;
  char *end = NULL;

  if (line == NULL || sample == NULL || strlen(line) >= sizeof(local)) {
    return false;
  }
  strcpy(local, line);

  for (uint32_t c = 0U; c < DIODEM_AI_CHANNELS; ++c) {
    sample[c] = strtof(cursor, &end);
    if (end == cursor) {
      return false;
    }
    if (c < DIODEM_AI_CHANNELS - 1U) {
      if (*end != ',') {
        return false;
      }
      cursor = end + 1;
    } else if (*end != '\0') {
      return false;
    }
  }
  return true;
}

static void print_result(uint32_t seq, DiodeMStreamStatus status, const DiodeMAiOutput *output) {
  char buffer[256];
  if (status == DIODEM_STREAM_WARMUP) {
    snprintf(buffer, sizeof(buffer), "%lu,WARMUP,%lu/%u\r\n",
             (unsigned long)seq,
             (unsigned long)g_stream.samples_seen,
             (unsigned int)DIODEM_AI_WINDOW_SIZE);
    uart_write(buffer);
    return;
  }
  if (status != DIODEM_STREAM_OK || output == NULL) {
    snprintf(buffer, sizeof(buffer), "%lu,ERR\r\n", (unsigned long)seq);
    uart_write(buffer);
    return;
  }

  snprintf(buffer, sizeof(buffer),
           "%lu,OK,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%lu\r\n",
           (unsigned long)seq,
           output->values[0],
           output->values[1],
           output->values[2],
           output->values[3],
           output->values[4],
           output->values[5],
           (unsigned long)output->inference_us);
  uart_write(buffer);
}

static void MX_USART3_Polling_Init(void) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  uint32_t pclk = 0U;

  /* ST-LINK VCP on NUCLEO-H723ZG: USART3 TX=PD8, RX=PD9. */
  __HAL_RCC_USART3_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_USART3_FORCE_RESET();
  __HAL_RCC_USART3_RELEASE_RESET();

  GPIO_InitStruct.Pin = GPIO_PIN_8 | GPIO_PIN_9;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART3;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  USART3->CR1 = 0U;
  USART3->CR2 = 0U;
  USART3->CR3 = 0U;
#if defined(USART_PRESC_PRESCALER)
  USART3->PRESC = 0U;
#endif
  pclk = HAL_RCC_GetPCLK1Freq();
  if (pclk == 0U) {
    Error_Handler();
  }
  USART3->BRR = (pclk + (115200U / 2U)) / 115200U;
  USART3->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;

  for (uint32_t guard = 0U; guard < USART_READY_GUARD_LIMIT; ++guard) {
    if ((USART3->ISR & USART_ISR_TEACK) != 0U && (USART3->ISR & USART_ISR_REACK) != 0U) {
      return;
    }
  }
  Error_Handler();
}

static void MX_GPIO_Init(void) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);
}

static void Error_Handler(void) {
  fault_loop("ERR\r\n");
}

static void fault_loop(const char *code) {
  __disable_irq();
  if (code != NULL) {
    uart_write(code);
  }
  while (1) {
    HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_0);
    for (volatile uint32_t delay = 0U; delay < 800000U; ++delay) {
    }
  }
}

void HardFault_Handler(void) {
  fault_loop("HF\r\n");
}

void SysTick_Handler(void) {
  HAL_IncTick();
}

void MemManage_Handler(void) {
  fault_loop("MM\r\n");
}

void BusFault_Handler(void) {
  fault_loop("BF\r\n");
}

void UsageFault_Handler(void) {
  fault_loop("UF\r\n");
}
