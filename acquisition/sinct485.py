from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence


INPUT_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
DEFAULT_BAUDRATE = 9600
DEFAULT_SLAVE_ID = 0x50
DEFAULT_TIMEOUT_SEC = 0.05
MOTION_START_REGISTER = 0x34
MOTION_REGISTER_COUNT = 6
GRAVITY_MPS2 = 9.80665


class Sinct485Error(Exception):
    """Base error for SINCT-485 acquisition failures."""


class ModbusResponseError(Sinct485Error):
    """Raised when a Modbus RTU response is malformed or fails validation."""


def crc16_modbus(payload: bytes) -> int:
    """Return the CRC16-Modbus value for a payload, before little-endian packing."""
    crc = 0xFFFF
    for byte in payload:
        crc ^= int(byte)
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
            crc &= 0xFFFF
    return crc


def _append_crc(payload: bytes) -> bytes:
    crc = crc16_modbus(payload)
    return payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def build_read_registers_request(slave_id: int, start_register: int, register_count: int) -> bytes:
    """Build a Modbus RTU function-03 read-holding-registers request."""
    if not 0 <= int(slave_id) <= 0xFF:
        raise ValueError("slave_id must fit in one byte.")
    if not 0 <= int(start_register) <= 0xFFFF:
        raise ValueError("start_register must fit in two bytes.")
    if not 1 <= int(register_count) <= 0x7D:
        raise ValueError("register_count must be in the Modbus function-03 range 1..125.")
    payload = bytes(
        [
            int(slave_id),
            0x03,
            (int(start_register) >> 8) & 0xFF,
            int(start_register) & 0xFF,
            (int(register_count) >> 8) & 0xFF,
            int(register_count) & 0xFF,
        ]
    )
    return _append_crc(payload)


def _validate_crc(frame: bytes) -> None:
    if len(frame) < 4:
        raise ModbusResponseError(f"Response is too short for CRC validation: {len(frame)} bytes.")
    expected_crc = int.from_bytes(frame[-2:], byteorder="little", signed=False)
    actual_crc = crc16_modbus(frame[:-2])
    if actual_crc != expected_crc:
        raise ModbusResponseError(f"CRC mismatch: expected 0x{expected_crc:04x}, computed 0x{actual_crc:04x}.")


def _decode_signed_register(high_byte: int, low_byte: int) -> int:
    raw = (int(high_byte) << 8) | int(low_byte)
    if raw >= 0x8000:
        raw -= 0x10000
    return raw


def parse_read_registers_response(frame: bytes, slave_id: int, register_count: int) -> List[int]:
    """Validate and decode a Modbus RTU function-03 response into signed registers."""
    expected_byte_count = int(register_count) * 2
    expected_length = 3 + expected_byte_count + 2
    if len(frame) != expected_length:
        raise ModbusResponseError(f"Expected {expected_length} response bytes, got {len(frame)}.")
    _validate_crc(frame)

    if frame[0] != int(slave_id):
        raise ModbusResponseError(f"Unexpected slave id 0x{frame[0]:02x}; expected 0x{int(slave_id):02x}.")
    if frame[1] == 0x83:
        raise ModbusResponseError(f"Device returned Modbus exception code 0x{frame[2]:02x}.")
    if frame[1] != 0x03:
        raise ModbusResponseError(f"Unexpected function code 0x{frame[1]:02x}; expected 0x03.")
    if frame[2] != expected_byte_count:
        raise ModbusResponseError(f"Unexpected byte count {frame[2]}; expected {expected_byte_count}.")

    payload = frame[3:-2]
    return [
        _decode_signed_register(payload[index], payload[index + 1])
        for index in range(0, len(payload), 2)
    ]


def convert_motion_registers_to_sample(registers: Sequence[int]) -> Dict[str, float]:
    """Convert AX/AY/AZ/GX/GY/GZ registers to model-native physical units."""
    if len(registers) != MOTION_REGISTER_COUNT:
        raise ValueError(f"Expected {MOTION_REGISTER_COUNT} motion registers, got {len(registers)}.")
    acc_scale = 16.0 * GRAVITY_MPS2 / 32768.0
    gyr_scale = math.radians(2000.0) / 32768.0
    values = [
        float(registers[0]) * acc_scale,
        float(registers[1]) * acc_scale,
        float(registers[2]) * acc_scale,
        float(registers[3]) * gyr_scale,
        float(registers[4]) * gyr_scale,
        float(registers[5]) * gyr_scale,
    ]
    return {channel: value for channel, value in zip(INPUT_CHANNELS, values)}


@dataclass
class Sinct485Reader:
    """Minimal SINCT-485 Modbus RTU reader for motion registers."""

    port: str
    baudrate: int = DEFAULT_BAUDRATE
    slave_id: int = DEFAULT_SLAVE_ID
    timeout: float = DEFAULT_TIMEOUT_SEC
    serial_factory: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        factory = self.serial_factory
        if factory is None:
            try:
                import serial
            except ImportError as exc:  # pragma: no cover - depends on runtime environment
                raise Sinct485Error(
                    "pyserial is required for SINCT-485 acquisition. Install it with `pip install pyserial`."
                ) from exc
            factory = serial.Serial

        self._serial = factory(
            port=self.port,
            baudrate=int(self.baudrate),
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=float(self.timeout),
            write_timeout=float(self.timeout),
        )
        self._request = build_read_registers_request(
            slave_id=int(self.slave_id),
            start_register=MOTION_START_REGISTER,
            register_count=MOTION_REGISTER_COUNT,
        )
        self._response_length = 3 + MOTION_REGISTER_COUNT * 2 + 2

    def read_registers(self) -> List[int]:
        """Read raw AX/AY/AZ/GX/GY/GZ registers from the device."""
        if hasattr(self._serial, "reset_input_buffer"):
            self._serial.reset_input_buffer()
        self._serial.write(self._request)
        if hasattr(self._serial, "flush"):
            self._serial.flush()
        response = self._serial.read(self._response_length)
        if len(response) != self._response_length:
            raise ModbusResponseError(
                f"Timed out waiting for {self._response_length} response bytes; got {len(response)}."
            )
        return parse_read_registers_response(
            response,
            slave_id=int(self.slave_id),
            register_count=MOTION_REGISTER_COUNT,
        )

    def read_sample(self) -> Dict[str, float]:
        """Read one physical-unit sample in model input channel order."""
        return convert_motion_registers_to_sample(self.read_registers())

    def close(self) -> None:
        if hasattr(self, "_serial") and hasattr(self._serial, "close"):
            self._serial.close()

    def __enter__(self) -> "Sinct485Reader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
