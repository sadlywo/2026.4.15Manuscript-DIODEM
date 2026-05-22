import math
import unittest

from acquisition.sinct485 import (
    ModbusResponseError,
    build_read_registers_request,
    convert_motion_registers_to_sample,
    crc16_modbus,
    parse_read_registers_response,
)


def with_recomputed_crc(frame):
    payload = bytes(frame[:-2])
    crc = crc16_modbus(payload)
    return payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


class TestSinct485Protocol(unittest.TestCase):
    def test_crc_and_read_request_match_modbus_fixture(self):
        payload = bytes.fromhex("50 03 00 34 00 06")

        self.assertEqual(crc16_modbus(payload), 0x8789)
        self.assertEqual(
            build_read_registers_request(slave_id=0x50, start_register=0x34, register_count=6),
            bytes.fromhex("50 03 00 34 00 06 89 87"),
        )

    def test_parse_response_and_convert_units(self):
        response = bytes.fromhex("50 03 0c 00 00 00 00 08 00 40 00 c0 00 00 00 31 2a")

        registers = parse_read_registers_response(response, slave_id=0x50, register_count=6)
        self.assertEqual(registers, [0, 0, 2048, 16384, -16384, 0])

        sample = convert_motion_registers_to_sample(registers)
        self.assertAlmostEqual(sample["acc_x"], 0.0, places=6)
        self.assertAlmostEqual(sample["acc_y"], 0.0, places=6)
        self.assertAlmostEqual(sample["acc_z"], 9.80665, places=5)
        self.assertAlmostEqual(sample["gyr_x"], math.radians(1000.0), places=6)
        self.assertAlmostEqual(sample["gyr_y"], math.radians(-1000.0), places=6)
        self.assertAlmostEqual(sample["gyr_z"], 0.0, places=6)

    def test_parse_response_rejects_protocol_mismatches(self):
        good = bytearray.fromhex("50 03 0c 00 00 00 00 08 00 40 00 c0 00 00 00 31 2a")

        bad_crc = bytearray(good)
        bad_crc[-1] ^= 0x01
        with self.assertRaisesRegex(ModbusResponseError, "CRC"):
            parse_read_registers_response(bytes(bad_crc), slave_id=0x50, register_count=6)

        bad_slave = bytearray(good)
        bad_slave[0] = 0x51
        with self.assertRaisesRegex(ModbusResponseError, "slave"):
            parse_read_registers_response(with_recomputed_crc(bad_slave), slave_id=0x50, register_count=6)

        bad_count = bytearray(good)
        bad_count[2] = 10
        with self.assertRaisesRegex(ModbusResponseError, "byte count"):
            parse_read_registers_response(with_recomputed_crc(bad_count), slave_id=0x50, register_count=6)


if __name__ == "__main__":
    unittest.main()
