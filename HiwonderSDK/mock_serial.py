# mock_serial.py

class Serial:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.is_open = True
        print(f"Serial port {self.port} initialized with baud rate {self.baudrate}")

    def open(self):
        self.is_open = True
        print("Serial port opened.")

    def close(self):
        self.is_open = False
        print("Serial port closed.")

    def write(self, data):
        print(f"Writing data to {self.port}: {data}")

    def read(self, size=1):
        print(f"Reading {size} bytes from {self.port}")
        return b'\x00' * size

    def readline(self):
        print("Reading line from {self.port}")
        return b'\x00'

    @property
    def is_open(self):
        return self._is_open

    @is_open.setter
    def is_open(self, value):
        self._is_open = value
