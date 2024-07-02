class MockGPIO:
    BCM = 'BCM'
    BOARD = 'BOARD'
    OUT = 'OUT'
    IN = 'IN'
    LOW = 0
    HIGH = 1

    def __init__(self):
        self.setwarnings = lambda state: print(f"Setting warnings to {state}")

    def setmode(self, mode):
        print(f"Setting mode to {mode}")

    def setup(self, channel, mode):
        print(f"Setting up channel {channel} as {mode}")

    def output(self, channel, state):
        print(f"Setting channel {channel} to state {state}")

    def input(self, channel):
        print(f"Reading state from channel {channel}")
        return self.LOW

    def cleanup(self):
        print("Cleaning up GPIO")


GPIO = MockGPIO()
