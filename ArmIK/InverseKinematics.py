#!/usr/bin/env python3
# encoding: utf-8
# 4-DOF robotic arm inverse kinematics: Given the coordinates (X, Y, Z) and the pitch angle, calculate the rotation angle for each joint.
# 2020/07/20 Aiden

import logging
from math import *

# CRITICAL, ERROR, WARNING, INFO, DEBUG
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class IK:
    # Servo indices from bottom to top
    # Common parameters for the 4-DOF robotic arm
    l1 = 6.10  # Distance from the center of the base to the second servo axis (6.10 cm)
    l2 = 10.16  # Distance from the second servo to the third servo (10.16 cm)
    l3 = 9.64  # Distance from the third servo to the fourth servo (9.64 cm)
    l4 = 0.00  # This value is assigned based on the initialization choice

    # Specific parameters for the pump version
    l5 = 4.70  # Distance from the fourth servo to the point directly above the nozzle (4.70 cm)
    l6 = 4.46  # Distance from the point directly above the nozzle to the nozzle (4.46 cm)
    alpha = degrees(atan(l6 / l5))  # Calculate the angle between l5 and l6

    def __init__(self, arm_type):
        # Adapt parameters based on different gripper types
        self.arm_type = arm_type
        if self.arm_type == 'pump':  # If it is a pump version robotic arm
            self.l4 = sqrt(pow(self.l5, 2) + pow(self.l6, 2))  # Calculate the fourth link length for the pump version
        elif self.arm_type == 'arm':
            self.l4 = 16.65  # Distance from the fourth servo to the end of the robotic arm (16.65 cm)

    def setLinkLength(self, L1=l1, L2=l2, L3=l3, L4=l4, L5=l5, L6=l6):
        # Change the link length to adapt to robotic arms with the same structure but different lengths
        self.l1 = L1
        self.l2 = L2
        self.l3 = L3
        self.l4 = L4
        self.l5 = L5
        self.l6 = L6
        if self.arm_type == 'pump':
            self.l4 = sqrt(pow(self.l5, 2) + pow(self.l6, 2))
            self.alpha = degrees(atan(self.l6 / self.l5))

    def getLinkLength(self):
        # Get the current link length settings
        if self.arm_type == 'pump':
            return {"L1": self.l1, "L2": self.l2, "L3": self.l3, "L4": self.l4, "L5": self.l5, "L6": self.l6}
        else:
            return {"L1": self.l1, "L2": self.l2, "L3": self.l3, "L4": self.l4}

    def getRotationAngle(self, coordinate_data, Alpha):
        # Given coordinates and pitch angle, return the rotation angle for each joint, or False if there is no solution
        # coordinate_data is the end-effector coordinates in cm, passed as a tuple, e.g., (0, 5, 10)
        # Alpha is the angle between the end-effector and the horizontal plane, in degrees

        X, Y, Z = coordinate_data
        if self.arm_type == 'pump':
            Alpha -= self.alpha

        # Calculate the base rotation angle
        theta6 = degrees(atan2(Y, X))

        P_O = sqrt(X ** 2 + Y ** 2)  # Distance from the ground projection of the end-effector to the origin
        CD = self.l4 * cos(radians(Alpha))
        PD = self.l4 * sin(radians(Alpha))  # Positive when Alpha is positive, negative when Alpha is negative
        AF = P_O - CD
        CF = Z - self.l1 - PD
        AC = sqrt(AF ** 2 + CF ** 2)

        if round(CF, 4) < -self.l1:
            logger.debug('Height below 0, CF(%s) < l1(%s)', CF, -self.l1)
            return False
        if self.l2 + self.l3 < round(AC, 4):  # The sum of two sides is less than the third side
            logger.debug('Cannot form a linkage structure, l2(%s) + l3(%s) < AC(%s)', self.l2, self.l3, AC)
            return False

        # Calculate theta4
        cos_ABC = round(-(AC ** 2 - self.l2 ** 2 - self.l3 ** 2) / (2 * self.l2 * self.l3), 4)  # Cosine theorem
        if abs(cos_ABC) > 1:
            logger.debug('Cannot form a linkage structure, abs(cos_ABC(%s)) > 1', cos_ABC)
            return False
        ABC = acos(cos_ABC)  # Calculate the angle in radians
        theta4 = 180.0 - degrees(ABC)

        # Calculate theta5
        CAF = acos(AF / AC)
        cos_BAC = round((AC ** 2 + self.l2 ** 2 - self.l3 ** 2) / (2 * self.l2 * AC), 4)  # Cosine theorem
        if abs(cos_BAC) > 1:
            logger.debug('Cannot form a linkage structure, abs(cos_BAC(%s)) > 1', cos_BAC)
            return False
        theta5 = degrees(CAF * (1 if CF >= 0 else -1) + acos(cos_BAC))

        # Calculate theta3
        theta3 = Alpha - theta5 + theta4
        if self.arm_type == 'pump':
            theta3 += self.alpha

        return {"theta3": theta3, "theta4": theta4, "theta5": theta5,
                "theta6": theta6}  # Return the angles if there is a solution


if __name__ == '__main__':
    ik = IK('arm')
    ik.setLinkLength(L1=ik.l1 + 0.89, L4=ik.l4 - 0.3)
    print('Link lengths:', ik.getLinkLength())
    print(ik.getRotationAngle((0, 0, ik.l1 + ik.l2 + ik.l3 + ik.l4), 90))
