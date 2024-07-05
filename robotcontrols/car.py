import RPi.GPIO as GPIO
from time import sleep

# Define GPIO pins for motor 1 (left motor)
in1 = 24
in2 = 23
en = 25

# Define GPIO pins for motor 2 (right motor)
in3 = 22
in4 = 27
en2 = 17

# Default duration for turn and move left/right
t_duration = 1.2
t_sharp = 10  # Percentage decrease/increase in motor speeds

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

# Global variables to track current speeds
current_speed_p1 = 50
current_speed_p2 = 50

print("\n")
print("Default speed & direction of both motors: Medium & Forward.....")
print("Available commands:")
print("r - run both motors, s - stop both motors")
print("f - forward both motors, b - backward both motors")
print("sl - low speed, sm - medium speed, sh - high speed")
print("L - turn left, R - turn right")
print("l - move left, r - move right")
print("e - exit")
print("\n")

while True:
    x = input("Enter command: ")

    # Motor control
    if x == 'r':
        print("run")
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        print("forward both motors")
        x = 'z'

    elif x == 's':
        print("stop")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        x = 'z'

    elif x == 'f':
        print("move forward")
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        x = 'z'

    elif x == 'b':
        print("move backward")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        x = 'z'

    elif x == 'sl':
        print("low speed")
        p1.ChangeDutyCycle(25)
        p2.ChangeDutyCycle(25)
        current_speed_p1 = 25
        current_speed_p2 = 25
        x = 'z'

    elif x == 'sm':
        print("medium speed")
        p1.ChangeDutyCycle(50)
        p2.ChangeDutyCycle(50)
        current_speed_p1 = 50
        current_speed_p2 = 50
        x = 'z'

    elif x == 'sh':
        print("high speed")
        p1.ChangeDutyCycle(75)
        p2.ChangeDutyCycle(75)
        current_speed_p1 = 75
        current_speed_p2 = 75
        x = 'z'

    # Additional commands for specific movements
    elif x == 'L':
        print("turn left")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        sleep(t_duration)  # Use configured duration
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
        sleep(t_duration)  # Use configured duration
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        x = 'z'

    elif x == 'l':
        print("move left")
        global current_speed_p1, current_speed_p2

        # Adjust speeds
        new_speed_p1 = current_speed_p1 - (current_speed_p1 * t_sharp / 100)
        new_speed_p2 = current_speed_p2 + (current_speed_p2 * t_sharp / 100)
        p1.ChangeDutyCycle(new_speed_p1)
        p2.ChangeDutyCycle(new_speed_p2)

        sleep(t_duration)  # Use configured duration

        # Restore original speeds
        p1.ChangeDutyCycle(current_speed_p1)
        p2.ChangeDutyCycle(current_speed_p2)
        x = 'z'

    elif x == 'r':
        print("move right")
        # Similar logic as 'l', adjust speeds accordingly
        x = 'z'

    elif x == 'e':
        GPIO.cleanup()
        print("Exiting...")
        break

    else:
        print("<<<  wrong data  >>>")
        print("please enter the defined data to continue.....")
