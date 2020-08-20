import matplotlib.pyplot as plt
import numpy as np
import glob
import math
from moviepy.editor import VideoFileClip
import cv2
import os


team_members_names = ['اسلام محمد زكريا حسن', 'المروة وجدى عبدالعزيز المنشد', 'احمد نصر حماد عبدالحميد', 'عصام عماد درويش يوسف', 'اسلام ممدوح محمد محمد']
team_members_seatnumbers = ['2016170087', '2016170098', '2016170064', '2015170229', '2014170090']


def list_images(img, cols= 2, rows= 4, cmap=None, operation = 'Figure'):
    """
    Display all test images at one figure.
        Parameters:
            img: List of numpy arrays.
            cols (Default = 2): Number of columns in the figure.
            rows (Default = 4): Number of rows in the figure.
            cmap (Default = None): Used to display gray images.
            operation (Default = 'Figure'): Used to display operation happened.
    """
    fig = plt.figure(figsize=(10, 11))
    fig.suptitle(operation)

    for i, image in enumerate(img):
        plt.subplot(rows, cols, i+1)
        if len(image.shape) == 2:
            cmap = 'gray'
        plt.imshow(image, cmap = cmap)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    #made figure full screen for better visualization
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()

    plt.show()
    cv2.waitKey(0)

def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: Output of hough ransform.
    """
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)  # Y = (SLOPE) X + (INTERCEPT)
            length = np.sqrt( ( (y2 - y1) ** 2 ) + ( (x2 - x1) ** 2) )
            if slope >= 0:# (╱)
                left_lines.append( (slope, intercept) )
                left_weights.append( (length) )
            else:# (╲)
                right_lines.append( (slope, intercept) )
                right_weights.append( (length) )

    if len(left_weights) > 0:
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)
    else:#No Line
        left_lane = None

    if len(right_weights) > 0:
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)
    else:#No Line
        right_lane = None

    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """
    Convert the slope and intercept for each line to pixel points then return start and end of line.
        Parameters:
            y1: the line's starting y.
            y2: the line's end y.
            line: The slope and intercept of the line.
    """
    if line is None:#No line at this frame
        return None
    slope, intercept = line
    #GENERAL EQUATION --> Y = (slope) X + (intercept)
    x1 = int( (y1 - intercept) / slope )
    x2 = int( ( y2 - intercept ) / slope )
    y1 = int( y1 )
    y2 = int( y2 )
    return ((x1, y1), (x2, y2))

def create_full_lane(img, lines):
    """
    Create full line from pixel points.
        Parameters:
            img: The input image.
            lines: Output lines of Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = img.shape[0]#start of line draw on image
    y2 = y1 * 0.6#end of line draw on image
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lines_connected(img, lines, color= [255, 0, 0], thickness= 10):
    """
       Draw lines onto the input image.
           Parameters:
               img: The input image.
               lines: Output lines of Hough Transform.
               color (Default = red): Line color.
               thickness (Default = 8): Line thickness.
       """
    line_image = np.zeros_like(img)
    for line in lines:#tuble
        if line is not None:
            cv2.line(line_image, line[0], line[1], color, thickness)
    return cv2.addWeighted(img, 1, line_image, 1, 0)

def RGB_color_selection(image):
    """
    Apply color selection to RGB images by blackout all pixels except for white and yellow.
        Parameters:
            image: An numpy array.
    """
    # White color mask
    lower_threshold = np.uint8([200, 200, 200])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = np.uint8([175, 175, 0])
    upper_threshold = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def convert_rbg_to_grayscale(image):
    """
        Convert images to gray scale.
            Parameters:
                image: An numpy array.
        """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def HSV_color_selection(image):
    """
    Apply color selection to HSV images by blackout all pixels except for white and yellow.
        Parameters:
            image: An numpy array.
    """
    # Convert the input image to HSV
    converted_image = convert_rgb_to_hsv(image)

    # White color mask
    lower_threshold = np.uint8([0, 0, 210])
    upper_threshold = np.uint8([255, 30, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = np.uint8([18, 80, 80])
    upper_threshold = np.uint8([30, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

def convert_rgb_to_hsv(image):
    """
       Convert RGB images to HSV.
           Parameters:
               image: An numpy array.
       """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def remove_noise(image, kernel_size= 13):
    """
        Apply Gaussian filter to the input image.
            Parameters:
                image: An numpy array.
                kernel_size (Default = 13): The size of the Gaussian kernel. It must be an odd number (3, 5, 7, ...).
        """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def calculate_intensity_gradient(img):
    '''
        Finding the intensity gradient of the image using 5*5 soble filter.
            Parameters:
                img: An numby array.
    '''
    xGradient = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize=5)
    yGradient = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize=5)

    return xGradient, yGradient

def non_maximum_suppression(img, magnitude , slope):
    '''
         Thin out the edges by iterate over all the points on the gradient intensity matrix and finds the pixels with the maximum.
            Parameters:
                img: An numby array.
                magnitude: The magnitude of the gradient
                slope: The slope of the gradient
    '''
    # getting the dimensions of the input image
    height, width = img.shape

    # Looping through every pixel of the grayscale
    for x in range(width):
        for y in range(height):
            grad_ang = slope[y, x]
            if abs(grad_ang) > 180:
                grad_ang = abs(grad_ang - 180)
            else:
                grad_ang = abs(grad_ang)

            # selecting the neighbours of the target pixel according to the gradient direction. In the x axis direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = x - 1, y
                neighb_2_x, neighb_2_y = x + 1, y

                # top right (diagnol-1) direction
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = x - 1, y - 1
                neighb_2_x, neighb_2_y = x + 1, y + 1

            # In y-axis direction
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = x, y - 1
                neighb_2_x, neighb_2_y = x, y + 1

            # top left (diagnol-2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = x - 1, y + 1
                neighb_2_x, neighb_2_y = x + 1, y - 1

            # Now it restarts the cycle
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = x - 1, y
                neighb_2_x, neighb_2_y = x + 1, y

                # Non-maximum suppression step
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if magnitude [y, x] < magnitude [neighb_1_y, neighb_1_x]:
                    magnitude [y, x] = 0
                    continue

            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if magnitude [y, x] < magnitude [neighb_2_y, neighb_2_x]:
                    magnitude [y, x] = 0

    return magnitude

def double_threshold_and_hysteresis(suppressd_img, ids, low_threshold, high_threshold):
    '''
    Identifying strong, weak, and non-relevant pixels. Based on the threshold results, the hysteresis transforms weak pixels into strong ones.
        Parameters:
            suppressd_img: Output of non maximum suppression.
            ids: An numpy array(1, 2).
            low_threshold: lowest value for pixel to include.
            high_threshold: highest value for pixel to include.
    '''
    height, width = suppressd_img.shape
    for x in range(width):
        for y in range(height):
            grad_mag = suppressd_img[y, x]

            if grad_mag < low_threshold:#non-relevant pixel
                suppressd_img[y, x] = 0
            elif grad_mag  >= low_threshold and grad_mag < high_threshold:#WEAK
                ids[y, x] = 1
                try:#weak edge--> hysteresis
                    if ((ids[x + 1, y - 1] == 2) or (ids[x + 1, y] == 2) or (ids[x + 1, y + 1] == 2)
                     or (ids[x, y - 1] == 2) or (ids[x, y + 1] == 2)
                     or (ids[x - 1, y - 1] == 2) or (suppressd_img[y - 1, y] == 2) or (suppressd_img[x - 1, y + 1] == 2)):
                        suppressd_img[x, y] = 150
                except IndexError as e:
                    pass
            else:#STRONG
                ids[y, x] = 2
    return suppressd_img

def detect_edges_canny(img, low_threshold= 50, high_threshold= 150):
    '''
        Apply Canny Edge Detection algorithm to the input image.
            Parameters:
                img: An numpy array.
                low_threshold (Default = 50).
                high_threshold (Default = 150).
    '''
    '''
        STEPS:
            Gaussian filter: done before.
            Finding the intensity gradient of the image.
            Non-maximum suppression.
            Double threshold.
            Edge tracking by hysteresis.
    '''
    # Calculating the gradients
    gx, gy = calculate_intensity_gradient(img)

        # Conversion of Cartesian coordinates to polar
    magnitude, slope = cv2.cartToPolar(gx, gy, angleInDegrees=True) # GET: magnitude= sqrt(gx^2 + gy^2), Slope= arctan(gx/gy)

    # setting the minimum and maximum threshold for double thresholding
    mag_max = np.max(magnitude)
    if not low_threshold:
        low_threshold = mag_max * 0.1
    if not high_threshold:
        high_threshold = mag_max * 0.5

    # getting the dimensions of the input image
    height, width = img.shape

    #non maximum suppression
    suppressd_img  = non_maximum_suppression(img, magnitude , slope)

    ids = np.zeros_like(img)

    # double thresholding and hysteresis
    final_image = double_threshold_and_hysteresis(suppressd_img, ids, low_threshold, high_threshold)
    # finally returning the magnitude of gradients of edges
    return final_image .astype('uint8')

def mask_image(image):
    """
        Determine and cut the region of interest in the input image.
            Parameters:
                image: An numpy array.
        """
    mask = np.zeros_like(image)
    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #by trails that is the best result for getting region we want.
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def hough_transform(image):
    """
    Determine hough transform.
        Parameters:
            image: The output of Canny.
    """
    rho = 1              #Distance resolution of the accumulator in pixels.
    theta = np.pi/180    #Angle resolution of the accumulator in radians.
    threshold = 20       #Only lines that are greater than threshold will be returned.
    minLineLength = 20   #Line segments shorter than that are rejected.
    maxLineGap = 300     #Maximum allowed gap between points on the same line to link them
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)
    '''
    lines = cv2.HoughLines(image, rho, theta, threshold)
    line_co-ordinate = np.array()
    for item in lines:
        for r, theta in item:
            # Stores the value of cos(theta) in a
            a = np.cos(theta)
            # Stores the value of sin(theta) in b
            b = np.sin(theta)
            # x0 stores the value rcos(theta)
            x0 = a * r
            # y0 stores the value rsin(theta)
            y0 = b * r
            # stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            line_co-ordinate.add([x1, y1, x2, y2])
    return line_co-ordinate
    '''

def frame_processor(image):
    """
    Process the input frame to detect lane lines.
        Parameters:
            image: Single video frame.
    """
    # MAIN PARTS #

    # 2 convert to HSV
    color_select = HSV_color_selection(image)
    # 3 convert to Gray
    gray         = convert_rbg_to_grayscale(color_select)
    # 6 Apply noise remove (gaussian) to the masked gray image
    smooth       = remove_noise(gray)
    # 7 use canny detector and fine tune the thresholds (low and high values)
    edges        = detect_edges_canny(smooth)
    # 8 mask the image using the canny detector output
    region       = mask_image(edges)
    # 9 apply hough transform to find the lanes
    hough        = hough_transform(region)
    #draw line from hough onto image
    result       = draw_lines_connected(image, create_full_lane(image, hough))
    return result

def process_video(test_video, output_video):
    """
    Read input video stream and produce a video file with detected lane lines.
        Parameters:
            test_video: Input video.
            output_video: A video file with detected lane lines.
    """
    # 10 apply the pipeline you developed to the challenge videos
    input_video = VideoFileClip(os.path.join('Input_Videos', test_video), audio=False)
    processed = input_video.fl_image(frame_processor)
    if(os.path.exists('Output_Videos') == False):
        os.mkdir('Output_Videos')
    processed.write_videofile(os.path.join('Output_Videos', output_video), audio=False)

def test_process():

    # 1 read the image1
    test_images = [plt.imread(image) for image in glob.glob('test_images/*.jpg')]
    list_images(test_images, 2, 4, None, 'Original Images')

    list_images(list(map(RGB_color_selection, test_images)), 2, 4, None, 'Thershold White And Yellow FROM RGB')
    # 2 convert to HSV
    list_images(list(map(convert_rgb_to_hsv, test_images)), 2, 4, None, 'Convert RGB Images\n To HSV')
    # 4 Threshold HSV for Yellow and White (combine the two results together)
    list_images(list(map(HSV_color_selection, test_images)), 2, 4, None, 'Thershold White And Yellow\n FROM HSV')

    color_selected_images = list(map(HSV_color_selection, test_images))
    # 3 convert to Gray
    gray_images = list(map(convert_rbg_to_grayscale, color_selected_images))
    list_images(gray_images, 2, 4, None, 'Convert Image To Greyscale')
    # 6 Apply noise remove (gaussian) to the masked gray image
    blur_images = list(map(remove_noise, gray_images))
    list_images(blur_images, 2, 4, None, 'Image Blurring')
    # 7 use canny detector and fine tune the thresholds (low and high values)
    edge_detected_images = list(map(detect_edges_canny, blur_images))
    list_images(edge_detected_images, 2, 4, None, 'Apply Canny Edge\n Detection')
    # 8 mask the image using the canny detector output
    masked_image = list(map(mask_image, edge_detected_images))
    list_images(masked_image, 2, 4, None, 'Region Of Interest')
    # 9 apply hough transform to find the lanes
    hough_lines = list(map(hough_transform, masked_image))

    lane_images = []
    for image, lines in zip(test_images, hough_lines):
        lane_images.append(draw_lines_connected(image, create_full_lane(image, lines)))

    #Draw Lanes On image
    list_images(lane_images, 2, 4, None, 'Draw Lanes On Original Image')

def Run_White():
    process_video('White Lane.mp4', 'White Lane Result.mp4')

def Run_Yellow():
    process_video('Yello Lane.mp4', 'Yellow Lane Result.mp4')

def Run_Challenge():
    process_video('Challenge.mp4', 'Challenge Result.mp4')

'''test pipeline on some images from test videos'''
test_process()

#Run_White()
#Run_Yellow()
#Run_Challenge()