import time
import sys
import json
import redis

sys.path.append('/Users/ms/code/sense')
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Board as Board

# Initialize ArmIK and Board
AK = ArmIK()
default_p_values = [-90, -90, 0]
p_values = default_p_values

default_m2_values = [-90, 0, 900]
m2_values = default_m2_values

servo_grip = 500
servo_release = 200

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def move_arm(x, y, z):
    print(f"Moving arm to X: {x}, Y: {y}, Z: {z}")
    move_result = AK.setPitchRangeMoving((x, y, z), *p_values, 1000)
    return move_result

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
        grip_result = Board.setBusServoPulse(1, servo_angle, 500)
        time.sleep(1)
        return grip_result
    else:
        print("Unable to move arm to specified coordinates.")
        return "not gripped"

def release_object():
    result = Board.setBusServoPulse(1, servo_release, 500)
    time.sleep(1)
    return result

def rotate_gripper(angle):
    result = Board.setBusServoPulse(2, angle, 500)
    time.sleep(1)
    return result

def approach_gripper(value):
    global m2_values
    approach_grip_angle = round(float(value))
    m2_values[1] = min(max(approach_grip_angle, m2_values[0]), m2_values[2])
    print('set m2_values', *m2_values)
    result = Board.setBusServoPulse(3, m2_values[1], 1000)
    return result

def execute_command(command):
    command_id = command['id']
    command_to_execute = command['command']

    try:
        if command_to_execute == 'move_arm':
            result_status = move_arm(command['x'], command['y'], command['z'])
        elif command_to_execute == 'grip_object':
            result_status = grip_object(command['x'], command['y'], command['z'], command['width'])
        elif command_to_execute == 'release_object':
            result_status = release_object()
        elif command_to_execute == 'rotate_gripper':
            result_status = rotate_gripper(command['angle'])
        elif command_to_execute == 'approach_gripper':
            result_status = approach_gripper(command['value'])
        else:
            result_status = 'unknown_command'

        print(f"Command {command_id} execution result: {result_status}")
    except Exception as e:
        print(f"Error executing command '{command_to_execute}': {e}")

# Main loop to fetch and execute commands from Redis
while True:
    command_str = r.blpop('commands_queue', timeout=0)
    if command_str:
        command = json.loads(command_str[1])
        print(f"Received command: {command}")
        execute_command(command)
    else:
        print("No commands found in the queue.")
