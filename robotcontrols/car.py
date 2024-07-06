import RPi.GPIO as GPIO
from time import sleep

# Define GPIO pins for motor 1 (right motor)
in1 = 22
in2 = 27
en = 17

# Define GPIO pins for motor 2 (left motor)
in3 = 24
in4 = 23
en2 = 25

# Default duration for turn and move left/right
t_duration = 1.2
t_sharp = 50  # Percentage decrease/increase in motor speeds

# Initialize GPIO
GPIO.setmode(GPIO.BCM)

# Setup motor 1 GPIO pins (right motor)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(en, GPIO.OUT)
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
p1 = GPIO.PWM(en, 1000)
p1.start(50)  # Default to medium speed (50% duty cycle)

# Setup motor 2 GPIO pins (left motor)
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
print("f - forward both motors, b - backward both motors")
print("sl - low speed, sm - medium speed, sh - high speed, su - ultra speed")
print("L - turn left, R - turn right")
print("l - move left, r - move right")
print("e - exit")
print("\n")


# Function to move left
def move_left():
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


def move_right():
    global current_speed_p1, current_speed_p2

    # Adjust speeds
    new_speed_p1 = current_speed_p1 + (current_speed_p1 * t_sharp / 100)
    new_speed_p2 = current_speed_p2 - (current_speed_p2 * t_sharp / 100)
    p1.ChangeDutyCycle(new_speed_p1)
    p2.ChangeDutyCycle(new_speed_p2)

    sleep(t_duration)  # Use configured duration

    # Restore original speeds
    p1.ChangeDutyCycle(current_speed_p1)
    p2.ChangeDutyCycle(current_speed_p2)


# Main loop for user input
while True:
    x = input("Enter command: ")

    # Motor control
    if x == 'f':
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

    elif x == 'su':
        print("ultra speed")
        p1.ChangeDutyCycle(100)
        p2.ChangeDutyCycle(100)
        current_speed_p1 = 100
        current_speed_p2 = 100
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
        move_left()
        x = 'z'

    elif x == 'r':
        print("move right")
        move_right()
        x = 'z'

    elif x == 'e':
        GPIO.cleanup()
        print("Exiting...")
        break

    else:
        print("<<<  wrong data  >>>")
        print("please enter the defined data to continue.....")
