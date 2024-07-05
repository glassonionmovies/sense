import RPi.GPIO as GPIO
from time import sleep

# Define GPIO pins for motor 1
in1 = 24
in2 = 23
en = 25

# Define GPIO pins for motor 2
in3 = 22
in4 = 27
en2 = 17

temp1 = 1
temp2 = 1

# Initialize GPIO
GPIO.setmode(GPIO.BCM)

# Setup motor 1 GPIO pins
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(en, GPIO.OUT)
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
p1 = GPIO.PWM(en, 1000)
p1.start(50)  # Default to medium speed (50% duty cycle)

# Setup motor 2 GPIO pins
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(en2, GPIO.OUT)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)
p2 = GPIO.PWM(en2, 1000)
p2.start(50)  # Default to medium speed (50% duty cycle)

print("\n")
print("Default speed & direction of both motors: Medium & Forward.....")
print("Available commands:")
print("r - run both motors, s - stop both motors")
print("f - forward both motors, b - backward both motors")
print("l - low speed, m - medium speed, h - high speed")
print("L - turn left, R - turn right,")
print("l - move left, r - move right,")
print("F - move forward, B - move backward, e - exit")
print("\n")

while (1):
    x = input()

    # Motor 1 control
    if x == 'r':
        print("run")
        if temp1 == 1:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
            print("forward motor 1")
            x = 'z'
        else:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
            print("backward motor 1")
            x = 'z'

    elif x == 's':
        print("stop")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        x = 'z'

    elif x == 'f':
        print("forward")
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        temp1 = 1
        x = 'z'

    elif x == 'b':
        print("backward")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        temp1 = 0
        x = 'z'

    elif x == 'l':
        print("low speed")
        p1.ChangeDutyCycle(25)
        x = 'z'

    elif x == 'm':
        print("medium speed")
        p1.ChangeDutyCycle(50)
        x = 'z'

    elif x == 'h':
        print("high speed")
        p1.ChangeDutyCycle(75)
        x = 'z'

    # Motor 2 control
    elif x == 'u':
        print("run")
        if temp2 == 1:
            GPIO.output(in3, GPIO.HIGH)
            GPIO.output(in4, GPIO.LOW)
            print("forward motor 2")
            x = 'z'
        else:
            GPIO.output(in3, GPIO.LOW)
            GPIO.output(in4, GPIO.HIGH)
            print("backward motor 2")
            x = 'z'

    elif x == 't':
        print("stop")
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        x = 'z'

    elif x == 'v':
        print("forward")
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        temp2 = 1
        x = 'z'

    elif x == 'w':
        print("backward")
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        temp2 = 0
        x = 'z'

    elif x == 'x':
        print("low speed")
        p2.ChangeDutyCycle(25)
        x = 'z'

    elif x == 'y':
        print("medium speed")
        p2.ChangeDutyCycle(50)
        x = 'z'

    elif x == 'z':
        print("high speed")
        p2.ChangeDutyCycle(75)
        x = 'z'

    # Additional commands for specific movements
    elif x == 'L':
        print("turn left")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        sleep(3.2)  # Default duration for turning left
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        x = 'z'

    elif x == 'R':
        print("turn right")
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        sleep(3.2)  # Default duration for turning right
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        x = 'z'

    elif x == 'l':
        print("move left")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        p2.ChangeDutyCycle(75)  # High speed for motor 2
        sleep(3.2)  # Default duration for moving left
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        x = 'z'

    elif x == 'r':
        print("move right")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        p1.ChangeDutyCycle(75)  # High speed for motor 1
        sleep(3.2)  # Default duration for moving right
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        x = 'z'

    elif x == 'F':
        print("move forward")
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        x = 'z'

    elif x == 'B':
        print("move backward")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        x = 'z'

    elif x == 'e':
        GPIO.cleanup()
        break

    else:
        print("<<<  wrong data  >>>")
        print("please enter the defined data to continue.....")
