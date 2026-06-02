/**
  ******************************************************************************
  * @file    tcn_causal.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-06-01T17:59:08+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "tcn_causal.h"
#include "tcn_causal_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_tcn_causal
 
#undef AI_TCN_CAUSAL_MODEL_SIGNATURE
#define AI_TCN_CAUSAL_MODEL_SIGNATURE     "0x2a141cc5798172a1913fe156e9472f7c"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-06-01T17:59:08+0800"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_TCN_CAUSAL_N_BATCHES
#define AI_TCN_CAUSAL_N_BATCHES         (1)

static ai_ptr g_tcn_causal_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_tcn_causal_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  imu_window_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 384, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _input_projection_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4224, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _activation_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _Concat_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4224, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _activation_1_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _Add_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _Concat_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4352, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_1_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _activation_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _Concat_3_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4352, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_1_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _activation_3_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _Add_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _Concat_4_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _activation_4_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _Concat_5_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _activation_5_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _Add_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _Concat_6_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5120, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_3_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _activation_6_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _Concat_7_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5120, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_3_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _activation_7_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _Add_3_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _latent_projection_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _feature_shift_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _feature_gate_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  _Add_4_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _Add_5_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _residual_projection_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _Add_6_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  compensated_imu_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  _input_projection_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _input_projection_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_1_output_0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  block_3_zero_padding_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  block_2_zero_padding_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  block_1_zero_padding_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  block_0_zero_padding_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_1_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_1_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_1_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_1_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  _latent_projection_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  _latent_projection_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  _feature_shift_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  _feature_shift_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  _feature_gate_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  _feature_gate_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  _Add_4_output_0_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  _Add_4_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  _residual_projection_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  _residual_projection_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  _input_projection_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  _latent_projection_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  _feature_shift_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  _feature_gate_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  _residual_projection_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Add_1_output_0_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_1_output_0_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _Add_2_output_0_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_2_output_0_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _Add_3_output_0_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_3_output_0_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _Add_4_output_0_bias, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_4_output_0_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _Add_4_output_0_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_4_output_0_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _Add_4_output_0_scale, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_4_output_0_scale_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _Add_5_output_0_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_5_output_0_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _Add_6_output_0_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 64), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &_Add_6_output_0_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _Add_output_0_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_output_0_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _Concat_1_output_0_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 66), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Concat_1_output_0_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _Concat_2_output_0_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 68), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Concat_2_output_0_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _Concat_3_output_0_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 68), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Concat_3_output_0_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _Concat_4_output_0_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 72), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Concat_4_output_0_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _Concat_5_output_0_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 72), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Concat_5_output_0_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _Concat_6_output_0_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 80), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Concat_6_output_0_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _Concat_7_output_0_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 80), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Concat_7_output_0_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _Concat_output_0_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 66), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Concat_output_0_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_1_output_0, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_Constant_1_output_0_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _Mul_output_0_output, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Mul_output_0_output_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _Sigmoid_output_0_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Sigmoid_output_0_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _activation_1_Relu_output_0_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_1_Relu_output_0_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _activation_2_Relu_output_0_output, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_2_Relu_output_0_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _activation_3_Relu_output_0_output, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_3_Relu_output_0_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _activation_4_Relu_output_0_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_4_Relu_output_0_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _activation_5_Relu_output_0_output, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_5_Relu_output_0_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _activation_6_Relu_output_0_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_6_Relu_output_0_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _activation_7_Relu_output_0_output, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_7_Relu_output_0_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _activation_Relu_output_0_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_Relu_output_0_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_1_Conv_output_0_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_1_Conv_output_0_bias_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_1_Conv_output_0_output, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_1_Conv_output_0_output_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_1_Conv_output_0_weights, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_conv1_1_Conv_output_0_weights_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_2_Conv_output_0_bias, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_2_Conv_output_0_bias_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_2_Conv_output_0_output, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_2_Conv_output_0_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_2_Conv_output_0_weights, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_conv1_2_Conv_output_0_weights_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_3_Conv_output_0_bias, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_3_Conv_output_0_bias_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_3_Conv_output_0_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_3_Conv_output_0_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_3_Conv_output_0_weights, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_conv1_3_Conv_output_0_weights_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_Conv_output_0_bias, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_Conv_output_0_bias_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_Conv_output_0_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_Conv_output_0_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_Conv_output_0_scratch0, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 3), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_Conv_output_0_scratch0_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_Conv_output_0_weights, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_conv1_Conv_output_0_weights_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_1_Conv_output_0_bias, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv2_1_Conv_output_0_bias_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_1_Conv_output_0_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv2_1_Conv_output_0_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_1_Conv_output_0_weights, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_conv2_1_Conv_output_0_weights_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_2_Conv_output_0_bias, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv2_2_Conv_output_0_bias_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_2_Conv_output_0_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv2_2_Conv_output_0_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_2_Conv_output_0_weights, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_conv2_2_Conv_output_0_weights_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_3_Conv_output_0_bias, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv2_3_Conv_output_0_bias_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_3_Conv_output_0_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv2_3_Conv_output_0_output_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_3_Conv_output_0_weights, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_conv2_3_Conv_output_0_weights_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_Conv_output_0_bias, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv2_Conv_output_0_bias_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_Conv_output_0_output, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv2_Conv_output_0_output_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_Conv_output_0_scratch0, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 3), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv2_Conv_output_0_scratch0_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_Conv_output_0_weights, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_conv2_Conv_output_0_weights_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  _feature_gate_Conv_output_0_bias, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_feature_gate_Conv_output_0_bias_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  _feature_gate_Conv_output_0_output, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_feature_gate_Conv_output_0_output_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  _feature_gate_Conv_output_0_scratch0, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_feature_gate_Conv_output_0_scratch0_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  _feature_gate_Conv_output_0_weights, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 64), AI_STRIDE_INIT(4, 4, 32, 2048, 2048),
  1, &_feature_gate_Conv_output_0_weights_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  _feature_shift_Conv_output_0_bias, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_feature_shift_Conv_output_0_bias_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  _feature_shift_Conv_output_0_output, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_feature_shift_Conv_output_0_output_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  _feature_shift_Conv_output_0_scratch0, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_feature_shift_Conv_output_0_scratch0_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  _feature_shift_Conv_output_0_weights, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 64), AI_STRIDE_INIT(4, 4, 32, 2048, 2048),
  1, &_feature_shift_Conv_output_0_weights_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  _input_projection_Conv_output_0_bias, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_input_projection_Conv_output_0_bias_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  _input_projection_Conv_output_0_output, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_input_projection_Conv_output_0_output_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  _input_projection_Conv_output_0_scratch0, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &_input_projection_Conv_output_0_scratch0_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  _input_projection_Conv_output_0_weights, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 6, 1, 1, 64), AI_STRIDE_INIT(4, 4, 24, 1536, 1536),
  1, &_input_projection_Conv_output_0_weights_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  _latent_projection_Conv_output_0_bias, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_latent_projection_Conv_output_0_bias_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  _latent_projection_Conv_output_0_output, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 64), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_latent_projection_Conv_output_0_output_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  _latent_projection_Conv_output_0_scratch0, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_latent_projection_Conv_output_0_scratch0_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  _latent_projection_Conv_output_0_weights, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 8), AI_STRIDE_INIT(4, 4, 256, 2048, 2048),
  1, &_latent_projection_Conv_output_0_weights_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  _residual_projection_Conv_output_0_bias, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &_residual_projection_Conv_output_0_bias_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  _residual_projection_Conv_output_0_output, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 64), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &_residual_projection_Conv_output_0_output_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  _residual_projection_Conv_output_0_scratch0, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_residual_projection_Conv_output_0_scratch0_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  _residual_projection_Conv_output_0_weights, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 6), AI_STRIDE_INIT(4, 4, 256, 1536, 1536),
  1, &_residual_projection_Conv_output_0_weights_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  block_0_zero_padding, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 2), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &block_0_zero_padding_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  block_1_zero_padding, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &block_1_zero_padding_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  block_2_zero_padding, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 8), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &block_2_zero_padding_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  block_3_zero_padding, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 16), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &block_3_zero_padding_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  compensated_imu_output, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &compensated_imu_output_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  imu_window_output, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 64), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &imu_window_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  compensated_imu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Add_6_output_0_output, &_Constant_1_output_0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &compensated_imu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  compensated_imu_layer, 42,
  GATHER_TYPE, 0x0, NULL,
  gather, forward_gather,
  &compensated_imu_chain,
  NULL, &compensated_imu_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_6_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &imu_window_output, &_residual_projection_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_6_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_6_output_0_layer, 40,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_6_output_0_chain,
  NULL, &compensated_imu_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _residual_projection_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_5_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_residual_projection_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_residual_projection_Conv_output_0_weights, &_residual_projection_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_residual_projection_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _residual_projection_Conv_output_0_layer, 39,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_residual_projection_Conv_output_0_chain,
  NULL, &_Add_6_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_5_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Mul_output_0_output, &_feature_shift_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_5_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_5_output_0_layer, 38,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_5_output_0_chain,
  NULL, &_residual_projection_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Add_3_output_0_output, &_Add_4_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Mul_output_0_layer, 37,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Mul_output_0_chain,
  NULL, &_Add_5_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_4_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_4_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Add_4_output_0_scale, &_Add_4_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_4_output_0_layer, 36,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_Add_4_output_0_chain,
  NULL, &_Mul_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_feature_gate_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Sigmoid_output_0_layer, 33,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_Sigmoid_output_0_chain,
  NULL, &_Add_4_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _feature_gate_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_latent_projection_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_feature_gate_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_feature_gate_Conv_output_0_weights, &_feature_gate_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_feature_gate_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _feature_gate_Conv_output_0_layer, 32,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_feature_gate_Conv_output_0_chain,
  NULL, &_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _feature_shift_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_latent_projection_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_feature_shift_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_feature_shift_Conv_output_0_weights, &_feature_shift_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_feature_shift_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _feature_shift_Conv_output_0_layer, 34,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_feature_shift_Conv_output_0_chain,
  NULL, &_feature_gate_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _latent_projection_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_latent_projection_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_latent_projection_Conv_output_0_weights, &_latent_projection_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_latent_projection_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _latent_projection_Conv_output_0_layer, 31,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_latent_projection_Conv_output_0_chain,
  NULL, &_feature_shift_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_activation_7_Relu_output_0_output, &_Add_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_3_output_0_layer, 30,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_3_output_0_chain,
  NULL, &_latent_projection_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_7_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv2_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_7_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_7_Relu_output_0_layer, 29,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_7_Relu_output_0_chain,
  NULL, &_Add_3_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv2_3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_7_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv2_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv2_3_Conv_output_0_weights, &_conv2_3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _conv2_3_Conv_output_0_layer, 28,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_conv2_3_Conv_output_0_chain,
  NULL, &_activation_7_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 8), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Concat_7_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_3_zero_padding, &_activation_6_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_7_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Concat_7_output_0_layer, 27,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_Concat_7_output_0_chain,
  NULL, &_conv2_3_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_6_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_6_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_6_Relu_output_0_layer, 26,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_6_Relu_output_0_chain,
  NULL, &_Concat_7_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv1_3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_6_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv1_3_Conv_output_0_weights, &_conv1_3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _conv1_3_Conv_output_0_layer, 25,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_conv1_3_Conv_output_0_chain,
  NULL, &_activation_6_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 8), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Concat_6_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_3_zero_padding, &_Add_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_6_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Concat_6_output_0_layer, 24,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_Concat_6_output_0_chain,
  NULL, &_conv1_3_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_activation_5_Relu_output_0_output, &_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_2_output_0_layer, 23,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_2_output_0_chain,
  NULL, &_Concat_6_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_5_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv2_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_5_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_5_Relu_output_0_layer, 22,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_5_Relu_output_0_chain,
  NULL, &_Add_2_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv2_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_5_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv2_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv2_2_Conv_output_0_weights, &_conv2_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _conv2_2_Conv_output_0_layer, 21,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_conv2_2_Conv_output_0_chain,
  NULL, &_activation_5_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 4), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Concat_5_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_2_zero_padding, &_activation_4_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_5_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Concat_5_output_0_layer, 20,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_Concat_5_output_0_chain,
  NULL, &_conv2_2_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_4_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_4_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_4_Relu_output_0_layer, 19,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_4_Relu_output_0_chain,
  NULL, &_Concat_5_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv1_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_4_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv1_2_Conv_output_0_weights, &_conv1_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _conv1_2_Conv_output_0_layer, 18,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_conv1_2_Conv_output_0_chain,
  NULL, &_activation_4_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 4), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Concat_4_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_2_zero_padding, &_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_4_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Concat_4_output_0_layer, 17,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_Concat_4_output_0_chain,
  NULL, &_conv1_2_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_activation_3_Relu_output_0_output, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_1_output_0_layer, 16,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_1_output_0_chain,
  NULL, &_Concat_4_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_3_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv2_1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_3_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_3_Relu_output_0_layer, 15,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_3_Relu_output_0_chain,
  NULL, &_Add_1_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv2_1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv2_1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv2_1_Conv_output_0_weights, &_conv2_1_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _conv2_1_Conv_output_0_layer, 14,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_conv2_1_Conv_output_0_chain,
  NULL, &_activation_3_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 2), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Concat_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_1_zero_padding, &_activation_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Concat_3_output_0_layer, 13,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_Concat_3_output_0_chain,
  NULL, &_conv2_1_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_2_Relu_output_0_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_2_Relu_output_0_chain,
  NULL, &_Concat_3_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv1_1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv1_1_Conv_output_0_weights, &_conv1_1_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _conv1_1_Conv_output_0_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_conv1_1_Conv_output_0_chain,
  NULL, &_activation_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 2), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Concat_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_1_zero_padding, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Concat_2_output_0_layer, 10,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_Concat_2_output_0_chain,
  NULL, &_conv1_1_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_activation_1_Relu_output_0_output, &_input_projection_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_output_0_layer, 9,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_output_0_chain,
  NULL, &_Concat_2_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_1_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_1_Relu_output_0_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_1_Relu_output_0_chain,
  NULL, &_Add_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv2_Conv_output_0_weights, &_conv2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_conv2_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _conv2_Conv_output_0_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_conv2_Conv_output_0_chain,
  NULL, &_activation_1_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Concat_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_0_zero_padding, &_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Concat_1_output_0_layer, 6,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_Concat_1_output_0_chain,
  NULL, &_conv2_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_Relu_output_0_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_Relu_output_0_chain,
  NULL, &_Concat_1_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv1_Conv_output_0_weights, &_conv1_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_conv1_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _conv1_Conv_output_0_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_conv1_Conv_output_0_chain,
  NULL, &_activation_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_0_zero_padding, &_input_projection_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Concat_output_0_layer, 3,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_Concat_output_0_chain,
  NULL, &_conv1_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _input_projection_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &imu_window_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_input_projection_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_input_projection_Conv_output_0_weights, &_input_projection_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_input_projection_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _input_projection_Conv_output_0_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_input_projection_Conv_output_0_chain,
  NULL, &_Concat_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 413500, 1, 1),
    413500, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 54784, 1, 1),
    54784, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_TCN_CAUSAL_IN_NUM, &imu_window_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_TCN_CAUSAL_OUT_NUM, &compensated_imu_output),
  &_input_projection_Conv_output_0_layer, 0xba463569, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 413500, 1, 1),
      413500, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 54784, 1, 1),
      54784, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_TCN_CAUSAL_IN_NUM, &imu_window_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_TCN_CAUSAL_OUT_NUM, &compensated_imu_output),
  &_input_projection_Conv_output_0_layer, 0xba463569, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool tcn_causal_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_tcn_causal_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    imu_window_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 36864);
    imu_window_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 36864);
    _input_projection_Conv_output_0_scratch0_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _input_projection_Conv_output_0_scratch0_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _input_projection_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 20480);
    _input_projection_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 20480);
    _Concat_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 3584);
    _Concat_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 3584);
    _conv1_Conv_output_0_scratch0_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _conv1_Conv_output_0_scratch0_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _conv1_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 3328);
    _conv1_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 3328);
    _activation_Relu_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _activation_Relu_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _Concat_1_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 3584);
    _Concat_1_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 3584);
    _conv2_Conv_output_0_scratch0_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 54016);
    _conv2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 54016);
    _conv2_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 3328);
    _conv2_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 3328);
    _activation_1_Relu_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 3328);
    _activation_1_Relu_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 3328);
    _Add_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _Add_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _Concat_2_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 19456);
    _Concat_2_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 19456);
    _conv1_1_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 3072);
    _conv1_1_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 3072);
    _activation_2_Relu_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 3072);
    _activation_2_Relu_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 3072);
    _Concat_3_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 19456);
    _Concat_3_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 19456);
    _conv2_1_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 3072);
    _conv2_1_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 3072);
    _activation_3_Relu_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 3072);
    _activation_3_Relu_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 3072);
    _Add_1_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _Add_1_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _Concat_4_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 3072);
    _Concat_4_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 3072);
    _conv1_2_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 2816);
    _conv1_2_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 2816);
    _activation_4_Relu_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 20480);
    _activation_4_Relu_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 20480);
    _Concat_5_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 2048);
    _Concat_5_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 2048);
    _conv2_2_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 20480);
    _conv2_2_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 20480);
    _activation_5_Relu_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 2048);
    _activation_5_Relu_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 2048);
    _Add_2_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _Add_2_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _Concat_6_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 2048);
    _Concat_6_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 2048);
    _conv1_3_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 1792);
    _conv1_3_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 1792);
    _activation_6_Relu_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 20480);
    _activation_6_Relu_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 20480);
    _Concat_7_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _Concat_7_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _conv2_3_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 20480);
    _conv2_3_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 20480);
    _activation_7_Relu_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _activation_7_Relu_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _Add_3_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 16384);
    _Add_3_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 16384);
    _latent_projection_Conv_output_0_scratch0_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _latent_projection_Conv_output_0_scratch0_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _latent_projection_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 32768);
    _latent_projection_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 32768);
    _feature_shift_Conv_output_0_scratch0_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _feature_shift_Conv_output_0_scratch0_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _feature_shift_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _feature_shift_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 38400);
    _feature_gate_Conv_output_0_scratch0_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 34816);
    _feature_gate_Conv_output_0_scratch0_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 34816);
    _feature_gate_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _feature_gate_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _Sigmoid_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _Sigmoid_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _Add_4_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _Add_4_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _Mul_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 16384);
    _Mul_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 16384);
    _Add_5_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _Add_5_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _residual_projection_Conv_output_0_scratch0_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 16384);
    _residual_projection_Conv_output_0_scratch0_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 16384);
    _residual_projection_Conv_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 16640);
    _residual_projection_Conv_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 16640);
    _Add_6_output_0_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    _Add_6_output_0_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 0);
    compensated_imu_output_array.data = AI_PTR(g_tcn_causal_activations_map[0] + 1536);
    compensated_imu_output_array.data_start = AI_PTR(g_tcn_causal_activations_map[0] + 1536);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool tcn_causal_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_tcn_causal_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _input_projection_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _input_projection_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 0);
    _input_projection_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 0);
    _input_projection_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _input_projection_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 1536);
    _input_projection_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 1536);
    _Constant_1_output_0_array.format |= AI_FMT_FLAG_CONST;
    _Constant_1_output_0_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 1792);
    _Constant_1_output_0_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 1792);
    block_3_zero_padding_array.format |= AI_FMT_FLAG_CONST;
    block_3_zero_padding_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 1796);
    block_3_zero_padding_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 1796);
    block_2_zero_padding_array.format |= AI_FMT_FLAG_CONST;
    block_2_zero_padding_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 5892);
    block_2_zero_padding_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 5892);
    block_1_zero_padding_array.format |= AI_FMT_FLAG_CONST;
    block_1_zero_padding_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 7940);
    block_1_zero_padding_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 7940);
    block_0_zero_padding_array.format |= AI_FMT_FLAG_CONST;
    block_0_zero_padding_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 8964);
    block_0_zero_padding_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 8964);
    _conv1_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv1_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 9476);
    _conv1_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 9476);
    _conv1_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv1_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 58628);
    _conv1_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 58628);
    _conv2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv2_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 58884);
    _conv2_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 58884);
    _conv2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv2_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 108036);
    _conv2_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 108036);
    _conv1_1_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv1_1_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 108292);
    _conv1_1_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 108292);
    _conv1_1_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv1_1_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 157444);
    _conv1_1_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 157444);
    _conv2_1_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv2_1_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 157700);
    _conv2_1_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 157700);
    _conv2_1_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv2_1_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 206852);
    _conv2_1_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 206852);
    _conv1_2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv1_2_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 207108);
    _conv1_2_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 207108);
    _conv1_2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv1_2_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 256260);
    _conv1_2_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 256260);
    _conv2_2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv2_2_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 256516);
    _conv2_2_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 256516);
    _conv2_2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv2_2_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 305668);
    _conv2_2_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 305668);
    _conv1_3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv1_3_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 305924);
    _conv1_3_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 305924);
    _conv1_3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv1_3_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 355076);
    _conv1_3_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 355076);
    _conv2_3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv2_3_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 355332);
    _conv2_3_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 355332);
    _conv2_3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv2_3_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 404484);
    _conv2_3_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 404484);
    _latent_projection_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _latent_projection_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 404740);
    _latent_projection_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 404740);
    _latent_projection_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _latent_projection_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 406788);
    _latent_projection_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 406788);
    _feature_shift_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _feature_shift_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 406820);
    _feature_shift_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 406820);
    _feature_shift_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _feature_shift_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 408868);
    _feature_shift_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 408868);
    _feature_gate_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _feature_gate_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 409124);
    _feature_gate_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 409124);
    _feature_gate_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _feature_gate_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 411172);
    _feature_gate_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 411172);
    _Add_4_output_0_scale_array.format |= AI_FMT_FLAG_CONST;
    _Add_4_output_0_scale_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 411428);
    _Add_4_output_0_scale_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 411428);
    _Add_4_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _Add_4_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 411684);
    _Add_4_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 411684);
    _residual_projection_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _residual_projection_Conv_output_0_weights_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 411940);
    _residual_projection_Conv_output_0_weights_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 411940);
    _residual_projection_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _residual_projection_Conv_output_0_bias_array.data = AI_PTR(g_tcn_causal_weights_map[0] + 413476);
    _residual_projection_Conv_output_0_bias_array.data_start = AI_PTR(g_tcn_causal_weights_map[0] + 413476);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_tcn_causal_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_TCN_CAUSAL_MODEL_NAME,
      .model_signature   = AI_TCN_CAUSAL_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 6546510,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xba463569,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_tcn_causal_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_TCN_CAUSAL_MODEL_NAME,
      .model_signature   = AI_TCN_CAUSAL_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 6546510,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xba463569,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_tcn_causal_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_tcn_causal_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_tcn_causal_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_tcn_causal_create(network, AI_TCN_CAUSAL_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_tcn_causal_data_params_get(&params) != true) {
    err = ai_tcn_causal_get_error(*network);
    return err;
  }
#if defined(AI_TCN_CAUSAL_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_TCN_CAUSAL_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_tcn_causal_init(*network, &params) != true) {
    err = ai_tcn_causal_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_tcn_causal_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_tcn_causal_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_tcn_causal_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_tcn_causal_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= tcn_causal_configure_weights(net_ctx, params);
  ok &= tcn_causal_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_tcn_causal_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_tcn_causal_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_TCN_CAUSAL_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

