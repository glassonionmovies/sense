import cv2
import numpy as np


def is_contour_outside_padding(contour, image_shape, padding_percentage=0.01):
    x, y, w, h = cv2.boundingRect(contour)
    image_height, image_width = image_shape[:2]
    padding_x = int(padding_percentage * image_width)
    padding_y = int(padding_percentage * image_height)

    if (x >= padding_x and y >= padding_y and
            (x + w) <= (image_width - padding_x) and
            (y + h) <= (image_height - padding_y)):
        return True
    return False


# Read the input color image
color_image = cv2.imread('/Users/ms/Downloads/a.jpg')

# Convert the color image to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred_image, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on proximity to the edge of the frame
valid_contours = [cnt for cnt in contours if is_contour_outside_padding(cnt, color_image.shape)]

# Draw bold red contour lines on the masked color image
color_image_with_contours = np.copy(color_image)
for contour in valid_contours:
    cv2.drawContours(color_image_with_contours, [contour], -1, (0, 0, 255), thickness=2)

# Display the color image with contours
cv2.imshow('Color Image with Contours', color_image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()