class MockPi:
    def __init__(self):
        self.setwarnings = lambda x: print("Setting warnings to", x)

    def set_mode(self, gpio, mode):
        print(f"Setting GPIO {gpio} to mode {mode}")

    def write(self, gpio, level):
        print(f"Setting GPIO {gpio} to level {level}")

    def read(self, gpio):
        print(f"Reading level from GPIO {gpio}")
        return 0

    def stop(self):
        print("Stopping mock pigpio")

def pi():
    return MockPi()

INPUT = "INPUT"
OUTPUT = "OUTPUT"

# Mock GPIO setwarnings method
GPIO = MockPi()
