#include <Arduino.h>

// Constants for motor control
const int COUNTS_PER_ROTATION = 1;   // Encoder counts per rotation

// Global variables for motor control
volatile long currentEncoderCountRight = 0;
volatile long targetEncoderCountRight = 0;
bool motorRunningRight = false;

volatile long currentEncoderCountLeft = 0;
volatile long targetEncoderCountLeft = 0;
bool motorRunningLeft = false;

// Motor speed control
int motorRunningSpeedRight = 255;  // Initial PWM duty cycle (0-255) for right motor
int motorRunningSpeedLeft = 255;   // Initial PWM duty cycle (0-255) for left motor

// Function Declarations
void encoderISRRight();
void encoderISRLeft();
void rotateMotorRight(int rotations);
void rotateMotorLeft(int rotations);
void rotateMotorsBoth(int rotations);
void stopMotorRight();
void stopMotorLeft();
void stopAllMotors();
void resetCounters();
int getRotations(String command);

void setup() {
  Serial.begin(9600);  // Initialize serial communication

  // Initialize Right Motor Pins
  pinMode(22, OUTPUT);  // IN1_PIN
  pinMode(23, OUTPUT);  // IN2_PIN
  pinMode(17, OUTPUT);  // PWM_PIN

  // Initialize Left Motor Pins
  pinMode(32, OUTPUT);  // IN3_PIN
  pinMode(33, OUTPUT);  // IN4_PIN
  pinMode(25, OUTPUT);  // PWM_PIN2

  // Initialize Right Motor Encoder Pins
  pinMode(18, INPUT);   // ENCODER_PIN_A
  pinMode(19, INPUT);   // ENCODER_PIN_B

  // Initialize Left Motor Encoder Pins
  pinMode(34, INPUT);   // ENCODER_PIN_A_LEFT
  pinMode(35, INPUT);   // ENCODER_PIN_B_LEFT

  // Attach interrupt handlers for encoder pins
  attachInterrupt(digitalPinToInterrupt(18), encoderISRRight, CHANGE);
  attachInterrupt(digitalPinToInterrupt(19), encoderISRRight, CHANGE);
  attachInterrupt(digitalPinToInterrupt(34), encoderISRLeft, CHANGE);
  attachInterrupt(digitalPinToInterrupt(35), encoderISRLeft, CHANGE);

  // Set initial motor direction and speed for both motors
  digitalWrite(22, HIGH);  // IN1_PIN
  digitalWrite(23, LOW);   // IN2_PIN
  analogWrite(17, 0);      // Stop right motor initially (PWM_PIN)

  digitalWrite(32, HIGH);  // IN3_PIN
  digitalWrite(33, LOW);   // IN4_PIN
  analogWrite(25, 0);      // Stop left motor initially (PWM_PIN2)
}

void loop() {
  handleSerialCommands();
}

void handleSerialCommands() {
  if (Serial.available() > 0) {
    String serialInput = Serial.readStringUntil('\n');
    serialInput.trim();  // Remove leading/trailing whitespace

    if (serialInput.startsWith("L:") || serialInput.startsWith("R:") || serialInput.startsWith("B:")) {
      int rotations = getRotations(serialInput);
      if (serialInput.startsWith("L:")) {
        rotateMotorLeft(rotations);
      } else if (serialInput.startsWith("R:")) {
        rotateMotorRight(rotations);
      } else if (serialInput.startsWith("B:")) {
        rotateMotorsBoth(rotations);
      }
    } else if (serialInput.equals("S")) {
      stopAllMotors();
    } else if (serialInput.equals("R")) {
      resetCounters();
    } else if (serialInput.startsWith("SL:")) {
      int speed = serialInput.substring(3).toInt();
      if (speed >= 0 && speed <= 255) {
        motorRunningSpeedLeft = speed;
      }
    } else if (serialInput.startsWith("SR:")) {
      int speed = serialInput.substring(3).toInt();
      if (speed >= 0 && speed <= 255) {
        motorRunningSpeedRight = speed;
      }
    } else if (serialInput.equals("E")) {
      // Print current encoder counts in summarized format
      Serial.print("LeftCount=");
      Serial.print(currentEncoderCountLeft);
      Serial.print(",RightCount=");
      Serial.println(currentEncoderCountRight);
    }
  }
}

void rotateMotorRight(int rotations) {
  targetEncoderCountRight = abs(rotations) * COUNTS_PER_ROTATION;
  currentEncoderCountRight = 0;
  motorRunningRight = true;

  if (rotations > 0) {
    digitalWrite(22, LOW);   // IN1_PIN (reversed)
    digitalWrite(23, HIGH);  // IN2_PIN (reversed)
  } else {
    digitalWrite(22, HIGH);  // IN1_PIN (reversed)
    digitalWrite(23, LOW);   // IN2_PIN (reversed)
  }
  analogWrite(17, motorRunningSpeedRight);  // PWM_PIN
}

void rotateMotorLeft(int rotations) {
  targetEncoderCountLeft = abs(rotations) * COUNTS_PER_ROTATION;
  currentEncoderCountLeft = 0;
  motorRunningLeft = true;

  if (rotations > 0) {
    digitalWrite(32, LOW);   // IN3_PIN (reversed)
    digitalWrite(33, HIGH);  // IN4_PIN (reversed)
  } else {
    digitalWrite(32, HIGH);  // IN3_PIN (reversed)
    digitalWrite(33, LOW);   // IN4_PIN (reversed)
  }
  analogWrite(25, motorRunningSpeedLeft);  // PWM_PIN2
}

void rotateMotorsBoth(int rotations) {
  targetEncoderCountRight = abs(rotations) * COUNTS_PER_ROTATION;
  currentEncoderCountRight = 0;
  motorRunningRight = true;
  targetEncoderCountLeft = abs(rotations) * COUNTS_PER_ROTATION;
  currentEncoderCountLeft = 0;
  motorRunningLeft = true;

  if (rotations > 0) {
    digitalWrite(22, LOW);   // IN1_PIN (reversed)
    digitalWrite(23, HIGH);  // IN2_PIN (reversed)
    digitalWrite(32, LOW);   // IN3_PIN (reversed)
    digitalWrite(33, HIGH);  // IN4_PIN (reversed)
  } else {
    digitalWrite(22, HIGH);  // IN1_PIN (reversed)
    digitalWrite(23, LOW);   // IN2_PIN (reversed)
    digitalWrite(32, HIGH);  // IN3_PIN (reversed)
    digitalWrite(33, LOW);   // IN4_PIN (reversed)
  }
  analogWrite(17, motorRunningSpeedRight);  // PWM_PIN
  analogWrite(25, motorRunningSpeedLeft);   // PWM_PIN2
}

void stopMotorRight() {
  analogWrite(17, 0);  // PWM_PIN
  motorRunningRight = false;
  //Serial.print("RightCount=");
  //Serial.println(currentEncoderCountRight);
}

void stopMotorLeft() {
  analogWrite(25, 0);  // PWM_PIN2
  motorRunningLeft = false;
  //Serial.print("LeftCount=");
  //Serial.println(currentEncoderCountLeft);
}

void stopAllMotors() {
  stopMotorRight();
  stopMotorLeft();
}

void resetCounters() {
  currentEncoderCountRight = 0;
  targetEncoderCountRight = 0;
  currentEncoderCountLeft = 0;
  targetEncoderCountLeft = 0;
}

int getRotations(String command) {
  int index = command.indexOf(':');  // Find the start of the rotations parameter
  if (index != -1) {
    index++;  // Move past ':'
    String rotationsStr = command.substring(index);  // Extract the rotations substring
    return rotationsStr.toInt();  // Convert substring to integer
  }
  return 0;  // Default to 0 rotations if parsing fails
}

void encoderISRRight() {
  static uint8_t lastStateRight = 0;
  uint8_t currentStateRight = (digitalRead(18) << 1) | digitalRead(19);  // ENCODER_PIN_A, ENCODER_PIN_B
  uint8_t stateRight = (lastStateRight << 2) | currentStateRight;

  // Handle encoder count based on quadrature encoder state
  switch (stateRight) {
    case 0b0001:
    case 0b0011:
    case 0b1100:
    case 0b1000:
      currentEncoderCountRight--;  // Decrement for forward rotation
      break;
    case 0b0010:
    case 0b0110:
    case 0b1101:
    case 0b1011:
      currentEncoderCountRight++;  // Increment for backward rotation
      break;
  }
  lastStateRight = currentStateRight;

  // Check if target encoder count has been reached
  if (motorRunningRight && abs(currentEncoderCountRight) >= targetEncoderCountRight) {
    stopMotorRight();
  }
}

void encoderISRLeft() {
  static uint8_t lastStateLeft = 0;
  uint8_t currentStateLeft = (digitalRead(34) << 1) | digitalRead(35);  // ENCODER_PIN_A_LEFT, ENCODER_PIN_B_LEFT
  uint8_t stateLeft = (lastStateLeft << 2) | currentStateLeft;

  // Handle encoder count based on quadrature encoder state
  switch (stateLeft) {
    case 0b0001:
    case 0b0011:
    case 0b1100:
    case 0b1000:
      currentEncoderCountLeft--;  // Decrement for forward rotation
      break;
    case 0b0010:
    case 0b0110:
    case 0b1101:
    case 0b1011:
      currentEncoderCountLeft++;  // Increment for backward rotation
      break;
  }
  lastStateLeft = currentStateLeft;

  // Check if target encoder count has been reached
  if (motorRunningLeft && abs(currentEncoderCountLeft) >= targetEncoderCountLeft) {
    stopMotorLeft();
  }
}
