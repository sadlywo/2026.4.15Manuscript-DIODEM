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


def test_firmware_contains_serial_hello_and_ai_placeholders():
    main_c = FIRMWARE_ROOT / "src" / "main.c"
    ai_stub = FIRMWARE_ROOT / "src" / "ai_inference_stub.c"
    readme = FIRMWARE_ROOT / "README.md"

    assert main_c.exists()
    assert ai_stub.exists()
    assert readme.exists()

    main_text = main_c.read_text(encoding="utf-8")
    assert "USART3" in main_text
    assert "PD8" in main_text
    assert "PD9" in main_text
    assert "BOOT_DIAG_V3" in main_text
    assert "UART_TX_GUARD_LIMIT" in main_text
    assert "DiodeM_Stream_Push" in main_text
