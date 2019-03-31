# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # �ѱ� �ּ������� �̰� �ؾ���
import cv2  # opencv ���
import numpy as np

# prior_left_line = None
# prior_right_line = None
# # prior_left_slope = None
# # prior_right_slope = None
# # cnt = 0
# i = 0

def grayscale(img):  # ����̹����� ��ȯ
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):  # Canny �˰���
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):  # ����þ� ����
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI ����
    mask = np.zeros_like(img)  # mask = img�� ���� ũ���� �� �̹���
    if len(img.shape) > 2:  # Color �̹���(3ä��)��� :
        color = color3
    else:  # ��� �̹���(1ä��)��� :
        color = color1
    # vertices�� ���� ����� �̷��� �ٰ����κ�(ROI �����κ�)�� color�� ä��
    cv2.fillPoly(mask, vertices, color)
    # �̹����� color�� ä���� ROI�� ��ħ
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):  # �� �׸���
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # ��ǥ�� �׸���
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # ���� ��ȯ 1 * np.pi / 180 30 10 20
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len
                            , maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    return lines

def weighted_img(img, initial_img, ��=1, ��=1., ��=0.):  # �� �̹��� operlap �ϱ�
    return cv2.addWeighted(initial_img, ��, img, ��, ��)

def get_fitline(img, f_lines):  # ��ǥ�� ���ϱ�
    lines = np.squeeze(f_lines)
    # if len(lines.shape) < 2:
    #     return false
    lines = lines.reshape(lines.shape[0] * 2, 2)
    rows, cols = img.shape[:2]
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    # print(output)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
    x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)
    result = [x1, y1, x2, y2]
    if x2 == x1:
        slope = 10000 * (y2 - y1)
    else:
        slope = (y2-y1)/(x2-x1)
    return result, slope


def detect(frame, prior_right_line, prior_left_line, left_fit_line, right_fit_line, i):
    # cap = cv2.VideoCapture('black5.mp4')
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    height, width = frame.shape[:2]
    # if cnt != 5:
    #     cnt = cnt + 1
    #     cv2.imshow('result', frame)
    #     continue
    # else:
    #     cnt = 0

    bgr = frame
    gray = grayscale(bgr)
    mark = np.copy(gray)
    thresholds = gray < 80
    mark[thresholds] = 120
    bgr = gaussian_blur(gray,5)
    bgr = canny(bgr,40,120)
    vertices = np.array(
            [[(50, height), (width / 2 - 85, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
            dtype=np.int32)
    ROI_img = region_of_interest(bgr, vertices)  # ROI ����
    line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)  # ���� ��ȯ
    line_arr = np.squeeze(line_arr)

    if len(line_arr.shape)<2:
        return False,0,0,0,0,0

    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
    #���� ���� ����
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]
    # ���� ���� ����
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]
    # ���͸��� ���� ������
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    temp = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    L_lines, R_lines = L_lines[:, None], R_lines[:, None]

    #shape���� �ʴ� �� ���� - 0,1�� �� �ȵ�
    if L_lines.shape[0]>1 and R_lines.shape[0]>1:
        # ����, ������ ���� ��ǥ�� ���ϱ�
        left_fit_line, left_slope = get_fitline(frame, L_lines)
        right_fit_line, right_slope = get_fitline(frame, R_lines)

        #�� or ������ �� ���� ���͸� �� ���Ⱑ ���ڱ� �ٲ�°� ����
        if abs(left_slope)<0.6 or left_slope>0:
            left_fit_line = None
        if abs(right_slope)<0.6 or right_slope<0:
            right_fit_line = None

        #���� ��Ż ����(���� ���� �� �� ��)�� �� �� ���� �������� ����
        if left_fit_line is None or right_fit_line is None:
            left_fit_line = prior_left_line
            right_fit_line = prior_right_line
            i = i + 1
            print(i)
            if i == 6:
                print("��������")
                i = 0
                return "change", prior_left_line, prior_right_line,left_fit_line, right_fit_line, i
        else:
            i = 0
            prior_left_line = left_fit_line
            prior_right_line = right_fit_line

    if prior_right_line == None or prior_left_line == None:
        return False,0,0,0,0,0

    # ��ǥ�� �׸���
    draw_fit_line(temp, left_fit_line)
    draw_fit_line(temp, right_fit_line)
    result = weighted_img(temp, frame)# ���� �̹����� ����� �� overlap
    cv2.imshow('result2', result)  # ��� �̹��� ���

    return "pass", prior_left_line, prior_right_line, left_fit_line, right_fit_line, i
    # result = weighted_img(temp, frame)# ���� �̹����� ����� �� overlap

