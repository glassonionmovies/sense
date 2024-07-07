# motor_control.py
RPi.GPIO as GPIO

from time import sleep, time

# Define GPIO pins for motor 1 (right motor)
in1 = 22
in2 = 27
en = 17

# Define GPIO pins for motor 2 (left motor)
in3 = 24
in4 = 23
en2 = 25

# Default duration for turn and move left/right
t_duration = 0.8
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

# Global variables to track original speeds
original_speed_p1 = 50
original_speed_p2 = 50

import math

def angle_from_y_axis(bx, by):
    if by == 0:
        by = 0.0001
    return math.atan(bx / by)

def calculate_radius(chord_length, theta):
    if theta == 0:
        raise ValueError("Theta cannot be zero")
    return chord_length / (2 * math.sin(theta / 2))

def find_radius_and_theta(bx, by):
    if bx == 0:
        bx = 0.0001
    if by == 0:
        by = 0.0001
    chord_length = math.sqrt(bx ** 2 + by ** 2)
    theta = angle_from_y_axis(bx, by)
    radius = calculate_radius(chord_length, theta)
    return radius, theta

def wheel_distances(R, theta, W):
    R_right = R - W / 2
    R_left = R + W / 2
    return R_right * theta, R_left * theta

def calculate_wheel_speeds(bx, by, W):
    R, theta = find_radius_and_theta(bx, by)
    distance_right, distance_left = wheel_distances(R, theta, W)
    max_speed = 100
    smaller_speed = max_speed * min(distance_right, distance_left) / max(distance_right, distance_left)
    if distance_left < distance_right:
        return smaller_speed, max_speed
    else:
        return max_speed, smaller_speed



# List to track motor activity
motor_activity = []
unique_id = 0
last_timestamp = time()

# Variables to track total rotations
total_rotation_left = 0
total_rotation_right = 0
direction = 1

print("\n")
print("Default speed & direction of both motors: Medium & Forward.....")
print("Available commands:")
print("f [speed] - forward both motors, b [speed] - backward both motors")
print("L - turn left, R - turn right")
print("l - move left, r - move right")
print("s - stop all motors")
print("e - exit")
print("\n")

# Function to log motor activity
def log_motor_activity(left_speed, right_speed):
    global unique_id, last_timestamp, total_rotation_left, total_rotation_right, direction

    current_timestamp = time()
    elapsed_time = current_timestamp - last_timestamp

    left_rotation = direction * left_speed * elapsed_time
    right_rotation = direction * right_speed * elapsed_time

    total_rotation_left += left_rotation
    total_rotation_right += right_rotation

    motor_activity.append([unique_id, current_timestamp, left_speed, right_speed, elapsed_time])
    unique_id += 1
    last_timestamp = current_timestamp
    print(motor_activity)

# Function to move left
def move_left():
    global original_speed_p1, original_speed_p2

    log_motor_activity(original_speed_p2, original_speed_p1)

    # Adjust speeds
    new_speed_p1 = max(0, original_speed_p1 - (original_speed_p1 * t_sharp / 100))
    new_speed_p2 = min(100, original_speed_p2 + (original_speed_p2 * t_sharp / 100))
    p1.ChangeDutyCycle(new_speed_p1)
    p2.ChangeDutyCycle(new_speed_p2)

    sleep(t_duration)  # Use configured duration
    log_motor_activity(new_speed_p1, new_speed_p2)
    # Restore original speeds
    p1.ChangeDutyCycle(original_speed_p1)
    p2.ChangeDutyCycle(original_speed_p2)
    log_motor_activity(original_speed_p2, original_speed_p1)

# Function to move right
def move_right():
    global original_speed_p1, original_speed_p2

    log_motor_activity(original_speed_p1, original_speed_p2)

    # Adjust speeds
    new_speed_p1 = min(100, original_speed_p1 + (original_speed_p1 * t_sharp / 100))
    new_speed_p2 = max(0, original_speed_p2 - (original_speed_p2 * t_sharp / 100))
    p1.ChangeDutyCycle(new_speed_p1)
    p2.ChangeDutyCycle(new_speed_p2)
    log_motor_activity(new_speed_p1, new_speed_p2)

    sleep(t_duration)  # Use configured duration
    log_motor_activity(original_speed_p1, original_speed_p2)
    # Restore original speeds
    p1.ChangeDutyCycle(original_speed_p1)
    p2.ChangeDutyCycle(original_speed_p2)
    log_motor_activity(original_speed_p1, original_speed_p1)

# Main loop for user input
running = True
while running:
    x = input("Enter command: ").split()

    if x[0] == 'f':
        direction = 1
        speed = int(x[1]) if len(x) > 1 else original_speed_p1
        print(f"move forward at speed {speed}")
        log_motor_activity(speed, speed)
        p1.ChangeDutyCycle(speed)
        p2.ChangeDutyCycle(speed)
        original_speed_p1 = speed
        original_speed_p2 = speed
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)

    elif x[0] == 'b':
        direction = -1
        speed = int(x[1]) if len(x) > 1 else original_speed_p1
        print(f"move backward at speed {speed}")
        log_motor_activity(speed, speed)
        p1.ChangeDutyCycle(speed)
        p2.ChangeDutyCycle(speed)
        original_speed_p1 = speed
        original_speed_p2 = speed
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)

    elif x[0] == 'L':
        print("turn left")
        log_motor_activity(original_speed_p2, original_speed_p1)
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        sleep(t_duration)  # Use configured duration
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)

    elif x[0] == 'R':
        print("turn right")
        log_motor_activity(original_speed_p1, original_speed_p2)
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        sleep(t_duration)  # Use configured duration
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)

    elif x[0] == 'l':
        print("move left")
        move_left()

    elif x[0] == 'r':
        print("move right")
        move_right()

    elif x[0] == 's':
        print("stop all motors")
        log_motor_activity(0, 0)
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)

    elif x[0] == 'e':
        log_motor_activity(0, 0)
        GPIO.cleanup()
        print("Exiting...")
        running = False

        print(f"Total rotation for left motor: {total_rotation_left}")
        print(f"Total rotation for right motor: {total_rotation_right}")

    else:
        print("<<<  wrong data  >>>")
        print("please enter the defined data to continue.....")
