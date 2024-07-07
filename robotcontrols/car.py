import RPi.GPIO as GPIO
import math
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
p1 = GPIO.PWM(en, 1000)

# Setup motor 2 GPIO pins (left motor)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(en2, GPIO.OUT)
p2 = GPIO.PWM(en2, 1000)

# Global variables to track original speeds
original_speed_p1 = 50
original_speed_p2 = 50

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

# Function to log motor activity
def log_motor_activity(left_speed, right_speed):
    current_timestamp = time()
    elapsed_time = current_timestamp - last_timestamp
    left_rotation = direction * left_speed * elapsed_time
    right_rotation = direction * right_speed * elapsed_time
    motor_activity.append([unique_id, current_timestamp, left_speed, right_speed, elapsed_time])
    print(motor_activity)

# Function to move motors to a given point
def move_to_given_point(bx, by, D):
    global original_speed_p1, original_speed_p2
    left_speed, right_speed = calculate_wheel_speeds(bx, by, W=13)

    p1.start(right_speed)  # Start right motor with specified duty cycle
    p2.start(left_speed)   # Start left motor with specified duty cycle

    # Set default direction (forward)
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)

    sleep(D)

    # Stop PWM signals
    p1.stop()
    p2.stop()

    # Cleanup GPIO
    GPIO.cleanup()

# Example usage with user inputs
try:
    bx = float(input("Enter bx coordinate: "))
    by = float(input("Enter by coordinate: "))
    D = float(input("Enter duration (in seconds): "))
    move_to_given_point(bx, by, D)

except ValueError as e:
    print("Error:", e)

except KeyboardInterrupt:
    print("\nUser interrupted the program.")

finally:
    GPIO.cleanup()
