from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_ROOT = PROJECT_ROOT / "firmware" / "platformio_nucleo_h723zg"


def test_platformio_project_is_isolated_and_targets_nucleo_h723zg():
    platformio_ini = FIRMWARE_ROOT / "platformio.ini"

    assert platformio_ini.exists()
    text = platformio_ini.read_text(encoding="utf-8")
    assert "[env:nucleo_h723zg]" in text
    assert "board = nucleo_h723zg" in text
    assert "framework = stm32cube" in text
    assert "upload_protocol = stlink" in text


def test_firmware_contains_serial_hello_and_ai_inference_entrypoints():
    main_c = FIRMWARE_ROOT / "src" / "main.c"
    ai_stub = FIRMWARE_ROOT / "src" / "ai_inference_stub.c"
    streaming_model = FIRMWARE_ROOT / "lib" / "streaming_tcn" / "include" / "streaming_tcn.h"
    streaming_weights = FIRMWARE_ROOT / "lib" / "streaming_tcn" / "src" / "streaming_tcn_weights.c"
    readme = FIRMWARE_ROOT / "README.md"

    assert main_c.exists()
    assert ai_stub.exists()
    assert streaming_model.exists()
    assert streaming_weights.exists()
    assert readme.exists()

    main_text = main_c.read_text(encoding="utf-8")
    assert "USART3" in main_text
    assert "PD8" in main_text
    assert "PD9" in main_text
    assert "BOOT_DIAG_V4" in main_text
    assert "SystemClock_Config" in main_text
    assert "RCC_SYSCLKSOURCE_PLLCLK" in main_text
    assert "PWR_REGULATOR_VOLTAGE_SCALE0" in main_text
    assert "SCB_EnableICache" in main_text
    assert "CLK,core=" in main_text
    assert "UART_TX_GUARD_LIMIT" in main_text
    assert "SysTick_Handler" in main_text
    assert "HAL_IncTick" in main_text
    assert "DiodeM_Stream_Push" in main_text

    ai_text = ai_stub.read_text(encoding="utf-8")
    assert "StreamingTcn_Init" in ai_text
    assert "StreamingTcn_RunStep" in ai_text
    assert "DiodeM_AI_RunStep" in ai_text
