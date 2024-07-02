import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

while True:
    # Fetch command from Redis
    command_str = r.blpop('commands_queue', timeout=0)
    if command_str:
        command = json.loads(command_str[1])
        command_id = command['id']
        command_to_execute = command['command']

        # Execute the command
        try:
            print(f"Executing command '{command_to_execute}'...")

            # Simulate execution result (pass or fail)
            if command_id % 3 == 0:
                command['status'] = 'pass'
            else:
                command['status'] = 'fail'
            print(f"Execution result: {command['status']}")
        except Exception as e:
            print(f"Error executing command: {e}")
            command['status'] = 'fail'

        # Update status in Redis
        r.set(f"command_status:{command_id}", json.dumps(command))
        print(f"Command execution status updated: {command}")
    else:
        print("No commands found in the queue.")
