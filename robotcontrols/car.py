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

GPIO.setmode(GPIO.BCM)

# Setup motor 1 GPIO pins
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(en, GPIO.OUT)
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
p1 = GPIO.PWM(en, 1000)
p1.start(25)

# Setup motor 2 GPIO pins
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(en2, GPIO.OUT)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)
p2 = GPIO.PWM(en2, 1000)
p2.start(25)

print("\n")
print("The default speed & direction of both motors are LOW & Forward.....")
print("Motor 1 controls: r-run s-stop f-forward b-backward l-low m-medium h-high")
print("Motor 2 controls: u-run t-stop v-forward w-backward x-low y-medium z-high e-exit")
print("\n")

while (1):
    x = input()

    # Motor 1 control
    if x == 'r':
        print("run motor 1")
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
        print("stop motor 1")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        x = 'z'

    elif x == 'f':
        print("forward motor 1")
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        temp1 = 1
        x = 'z'

    elif x == 'b':
        print("backward motor 1")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        temp1 = 0
        x = 'z'

    elif x == 'l':
        print("low speed motor 1")
        p1.ChangeDutyCycle(25)
        x = 'z'

    elif x == 'm':
        print("medium speed motor 1")
        p1.ChangeDutyCycle(50)
        x = 'z'

    elif x == 'h':
        print("high speed motor 1")
        p1.ChangeDutyCycle(75)
        x = 'z'

    # Motor 2 control
    elif x == 'u':
        print("run motor 2")
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
        print("stop motor 2")
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        x = 'z'

    elif x == 'v':
        print("forward motor 2")
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        temp2 = 1
        x = 'z'

    elif x == 'w':
        print("backward motor 2")
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        temp2 = 0
        x = 'z'

    elif x == 'x':
        print("low speed motor 2")
        p2.ChangeDutyCycle(25)
        x = 'z'

    elif x == 'y':
        print("medium speed motor 2")
        p2.ChangeDutyCycle(50)
        x = 'z'

    elif x == 'z':
        print("high speed motor 2")
        p2.ChangeDutyCycle(75)
        x = 'z'

    elif x == 'e':
        GPIO.cleanup()
        break

    else:
        print("<<<  wrong data  >>>")
        print("please enter the defined data to continue.....")
