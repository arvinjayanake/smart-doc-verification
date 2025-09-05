import cv2
import os

def crop_driving_licence_bounding_box(image_path, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    height, width = image.shape[:2]

    x_start = int(width * 0.15)  # 5% from start
    x_end = int(width * 0.75)  # 35% from left (width)
    y_start = int(height * 0.27)  # 15% from top
    y_end = int(height * 0.35)  # 25% from top(height)

    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    # Add label
    label = "Licence Number"
    cv2.putText(image, label, (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Output saved to: {output_path}")

    return image

def crop_nic_new_bounding_box(image_path, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    height, width = image.shape[:2]

    x_start = int(width * 0.15)  # 5% from start
    x_end = int(width * 0.75)  # 35% from left (width)
    y_start = int(height * 0.27)  # 15% from top
    y_end = int(height * 0.35)  # 25% from top(height)

    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    # Add label
    label = "NIC Number"
    cv2.putText(image, label, (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Output saved to: {output_path}")

    return image

def crop_nic_old_bounding_box(image_path, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate bounding box coordinates based on image dimensions
    # These ratios are estimated based on the typical layout described
    x_start = int(width * 0.15)  # 5% from start
    x_end = int(width * 0.75)  # 35% from left (width)
    y_start = int(height * 0.27)  # 15% from top
    y_end = int(height * 0.35)  # 25% from top(height)

    # Draw rectangle around NIC number area
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    # Add label
    label = "NIC Number"
    cv2.putText(image, label, (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Output saved to: {output_path}")

    return image