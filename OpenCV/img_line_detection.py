import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[0, 255, 255], thickness=3):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    return img

def pipeline(image):
    """
    An image processing pipeline which will output
    an image with the lane lines annotated.
    """

    height = image.shape[0]
    width = image.shape[1]
	
	
	
	
    """region_of_interest_vertices = [
        (0, height*2 / 3),
		(0, height),
		(width, height),
		(width, height*2/3),
        (width / 2, height / 2),
    ]"""
    """region_of_interest_vertices = [
		(0, height),
		(width, height),
        (width / 2, height / 2)
    ]"""
    region_of_interest_vertices = [
		(0, 0),
		(0, height),
		(width, height),
		(width, 0)
	]

    kernel = np.ones((4,4),np.float32)/25
    blurred_image = cv2.filter2D(image,-1,kernel)
	
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
	
    cannyed_image = cv2.Canny(gray_image, 100, 200)
 
    cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )
	
	
 
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
 
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    
    if not np.any(lines):
        return image
        
    for line in lines:
        for x1, y1, x2, y2 in line:
            if(x2 != x1):
                slope = float(y2 - y1) / (x2 - x1)
	    if math.fabs(slope) < 0.5:
		continue
	    if slope <= 0:
		left_line_x.extend([x1, x2])
		left_line_y.extend([y1, y2])
	    else:
		right_line_x.extend([x1, x2])
		right_line_y.extend([y1, y2])

    if len(left_line_x)==0 or len(right_line_x)==0:
        return image

    min_y = int(image.shape[0] * (3 / 5))
    max_y = int(image.shape[0])

    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
 
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
 
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
 
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    line_image = draw_lines(
        image,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=5,
    )

    return line_image

image = cv2.imread('road2.png')
img = pipeline(image)
cv2.imshow("test", img)
cv2.waitKey(10000)

"""from moviepy.editor import VideoFileClip
from IPython.display import HTML
white_output = 'solidWhiteRight_output.mp4'
clip1 = VideoFileClip("solidWhiteRight_input.mp4")
white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(white_output, audio=False)"""

# reading in an image

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    image = pipeline(frame)
    cv2.imshow("test", image)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
