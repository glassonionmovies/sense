import time
import sys
sys.path.append('/Users/ms/code/sense')
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Board as Board

AK = ArmIK()
default_p_values = [-90, -90, 0]
p_values = default_p_values

default_m2_values = [-90, 0, 900]
m2_values = default_m2_values

servo_grip = 500
servo_release = 200

def move_arm(x, y, z):
    print(f"Moving arm to X: {x}, Y: {y}, Z: {z}")
    move_result = AK.setPitchRangeMoving((x, y, z), *p_values, 1000)
    print(move_result)

def grip_object(x, y, z, width):
    release_object()
    print(f"Gripping object X: {x}, Y: {y}, Z: {z}, Object Width: {width}")
    move_result = AK.setPitchRangeMoving((x, y, z), *p_values, 1000)
    print(move_result)
    result = move_result
    if result:
        time.sleep(result[2] / 1000)
        servo_angle = 500 - int((width / 2) * 10)
        print('servo angle:', servo_angle)
        Board.setBusServoPulse(1, servo_angle, 500)
        time.sleep(1)
    else:
        print("Unable to move arm to specified coordinates.")

def release_object():
    Board.setBusServoPulse(1, servo_release, 500)
    time.sleep(1)

def rotate_gripper(angle):
    Board.setBusServoPulse(2, angle, 500)
    time.sleep(1)

def approach_gripper():
    Board.setBusServoPulse(3, m2_values[1], 1000)

def update_approach_angle(value):
    global m2_values
    approach_grip_angle = round(float(value))
    m2_values = [min_approach_angle.get(), approach_grip_angle, max_approach_angle.get()]
    print('set m2_values', *m2_values)
    update_approach_label()

def update_approach_label():
    angle = approach_grip_angle
    min_angle = min_approach_angle.get()
    max_angle = max_approach_angle.get()
    if angle < min_angle:
        angle = min_angle
    elif angle > max_angle:
        angle = max_angle
    print(angle)

# Example usage:
move_arm(10, 10, 5)
grip_object(10, 10, 5, 10)
rotate_gripper(90)
