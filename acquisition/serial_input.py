"""
Serial / UART Microcontroller ADC Stream Interface.
Reads digitized raw ADC voice samples streamed over USB-UART / COM ports
(e.g., STM32, ESP32, or Arduino ADC sampling the receiver audio output).
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


class SerialADCInterface:
    """
    Manages serial COM port communication for real-time ADC streams.
    """

    @staticmethod
    def list_available_ports() -> List[Dict[str, str]]:
        """
        Enumerates available serial COM ports.
        """
        if not HAS_SERIAL:
            return []

        ports = serial.tools.list_ports.comports()
        port_list = []
        for p in ports:
            port_list.append({
                "device": p.device,
                "description": p.description,
                "hwid": p.hwid
            })
        return port_list

    @staticmethod
    def read_samples_block(
        port_name: str,
        baud_rate: int = 115200,
        num_samples: int = 16000,
        timeout_sec: float = 2.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reads a block of raw 16-bit or 8-bit integer samples from the serial port.
        """
        if not HAS_SERIAL:
            return np.zeros(num_samples, dtype=np.float32), {"status": "pyserial not installed"}

        try:
            with serial.Serial(port_name, baud_rate, timeout=timeout_sec) as ser:
                # Read 2 bytes per 16-bit sample
                raw_bytes = ser.read(num_samples * 2)
                if len(raw_bytes) >= 2:
                    # Parse signed 16-bit little-endian
                    int_samples = np.frombuffer(raw_bytes, dtype=np.int16)
                    # Normalize to [-1.0, 1.0]
                    audio = int_samples.astype(np.float32) / 32768.0
                    status = f"Successfully read {len(audio)} samples from {port_name}"
                else:
                    audio = np.zeros(num_samples, dtype=np.float32)
                    status = "Timeout: No ADC data received on serial stream"
        except Exception as e:
            audio = np.zeros(num_samples, dtype=np.float32)
            status = f"Serial port connection error: {str(e)}"

        metrics = {
            "status": status,
            "port": port_name,
            "baud_rate": baud_rate,
            "samples_read": len(audio)
        }

        return audio, metrics
