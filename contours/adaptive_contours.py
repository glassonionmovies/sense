import cv2
import numpy as np

# Function to calculate length of a contour
def calculate_contour_length(contour):
    return cv2.arcLength(contour, True)

# Function to check if a contour is outside the padding area and the box
def is_contour_inside_box(contour, image_shape, box, padding_percentage=0.01):
    x, y, w, h = cv2.boundingRect(contour)
    image_height, image_width = image_shape[:2]
    padding_x = int(padding_percentage * image_width)
    padding_y = int(padding_percentage * image_height)

    if (x >= padding_x and y >= padding_y and
            (x + w) <= (image_width - padding_x) and
            (y + h) <= (image_height - padding_y) and
            box[0] <= x and box[1] <= y and box[2] >= (x + w) and box[3] >= (y + h)):
        return True
    return False

# Function to find the longest side of a polygon
def find_longest_side(points):
    max_length = 0
    longest_side = None
    for i in range(len(points)):
        length = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
        if length > max_length:
            max_length = length
            longest_side = (points[i], points[(i + 1) % len(points)])
    return longest_side

# Function to calculate the angle between two points
def calculate_angle(point1, point2):
    return np.arctan2(point2[1] - point1[1], point2[0] - point1[0]) * 180 / np.pi

# Function to mask the angle of the longest side of the contour
def mask_angle(frame, box):

    a=0.2
    box = [box[0]*(1-a), box[1]*(1-a), box[2]*(1+a), box[3]*(1+a)]

    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on proximity to the edge of the frame and the rectangular box
    valid_contours = [cnt for cnt in contours if is_contour_inside_box(cnt, frame.shape, box)]

    # Calculate length for each contour
    lengths = [calculate_contour_length(contour) for contour in valid_contours]

    # Find the index of the contour with maximum length
    if lengths:
        max_length_index = lengths.index(max(lengths))
        longest_contour = valid_contours[max_length_index]

        # Approximate the contour with a polygon
        epsilon = 0.01 * cv2.arcLength(longest_contour, True)
        approx = cv2.approxPolyDP(longest_contour, epsilon, True)

        # Find the longest side of the polygon
        longest_side = find_longest_side(approx[:, 0])

        # Draw the longest side of the polygon in yellow
        cv2.line(frame, tuple(longest_side[0]), tuple(longest_side[1]), (0, 255, 255), thickness=5)  # Yellow thick line

        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0),
                      thickness=2)  # Green box

        cv2.drawContours(frame, valid_contours, -1, (0, 0, 255), thickness=2)

        # Calculate the angle of the yellow line
        angle = calculate_angle(longest_side[0], longest_side[1])
        cv2.imshow('Color Image with Longest Contour and Longest Side', frame)
        print("Angle of the yellow line:", angle)
        return(angle)

    else:
        print("No contours found.")
        return(-1)

# Main function
def main():
    # Read the input color image
    color_image = cv2.imread('/Users/ms/Downloads/w.jpg')

    # Hardcoded box format [x_min, y_min, x_max, y_max]
    box = [399.98, 68.1, 601.75, 305.76]

    # Call the mask_angle function
    mask_angle(color_image, box)

    # Display the color image with the longest contour and polygon

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

