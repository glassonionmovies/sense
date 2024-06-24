# sense in robotics
Robotics with Transformers 


Collection of ideas to try/implement
1. Using checkerboard snapshot to get real-world coordinates. Note x.y,z of the arm. Get to that arm position, then get world coordinates of objects
2. Also identify first how robotic arm moves. It should move while its looking. That should be the angle for world coordinate.
3. Angle of camera is not needed because world coordinate can be obtained at #2
4. Rectification of left/right image is possible and implemented. But doesn't give accurate coordinate or distance. Need to calibrate stereo with rectified and then use rectified in owl.


To install pytorch on Jetson use the wheels from https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

Progress so far
1. Calibrate camera for new world coordinates projection matrix -> P1/P2 world and P1/P2 orig
2. OwlVit2 implemented -> Identifies the most confident object -> Box
3. FastSAM on box to get mask and find top right most mask point for each left and right image -> left_point, right_point
4. If either left or right mask point is not available use center of box -> left_center, right_center

Backlog OSAF
1. Integrate get angle on masked image
2. Based on angle, find object size - length and width and possibly height
3. In case of multiple masks, pick up the mask inside the box.
4. Filter mask to keep only inside box mask (may get solved with #3)
