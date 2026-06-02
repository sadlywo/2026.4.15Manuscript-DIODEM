/**
  ******************************************************************************
  * @file    tcn_causal_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-06-01T17:59:08+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef TCN_CAUSAL_DATA_PARAMS_H
#define TCN_CAUSAL_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_TCN_CAUSAL_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_tcn_causal_data_weights_params[1]))
*/

#define AI_TCN_CAUSAL_DATA_CONFIG               (NULL)


#define AI_TCN_CAUSAL_DATA_ACTIVATIONS_SIZES \
  { 54784, }
#define AI_TCN_CAUSAL_DATA_ACTIVATIONS_SIZE     (54784)
#define AI_TCN_CAUSAL_DATA_ACTIVATIONS_COUNT    (1)
#define AI_TCN_CAUSAL_DATA_ACTIVATION_1_SIZE    (54784)



#define AI_TCN_CAUSAL_DATA_WEIGHTS_SIZES \
  { 413500, }
#define AI_TCN_CAUSAL_DATA_WEIGHTS_SIZE         (413500)
#define AI_TCN_CAUSAL_DATA_WEIGHTS_COUNT        (1)
#define AI_TCN_CAUSAL_DATA_WEIGHT_1_SIZE        (413500)



#define AI_TCN_CAUSAL_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_tcn_causal_activations_table[1])

extern ai_handle g_tcn_causal_activations_table[1 + 2];



#define AI_TCN_CAUSAL_DATA_WEIGHTS_TABLE_GET() \
  (&g_tcn_causal_weights_table[1])

extern ai_handle g_tcn_causal_weights_table[1 + 2];


#endif    /* TCN_CAUSAL_DATA_PARAMS_H */
