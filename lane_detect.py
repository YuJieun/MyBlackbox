# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함
import cv2  # opencv 사용
import numpy as np

# prior_left_line = None
# prior_right_line = None
# # prior_left_slope = None
# # prior_right_slope = None
# # cnt = 0
# i = 0

def grayscale(img):  # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅
    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지
    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # 대표선 그리기
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환 1 * np.pi / 180 30 10 20
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len
                            , maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    return lines

def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_fitline(img, f_lines):  # 대표선 구하기
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
    ROI_img = region_of_interest(bgr, vertices)  # ROI 설정
    line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)  # 허프 변환
    line_arr = np.squeeze(line_arr)

    if len(line_arr.shape)<2:
        return False,0,0,0,0,0

    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
    #수평 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]
    # 수직 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]
    # 필터링된 직선 버리기
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    temp = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    L_lines, R_lines = L_lines[:, None], R_lines[:, None]

    #shape맞지 않는 것 제외 - 0,1일 때 안됨
    if L_lines.shape[0]>1 and R_lines.shape[0]>1:
        # 왼쪽, 오른쪽 각각 대표선 구하기
        left_fit_line, left_slope = get_fitline(frame, L_lines)
        right_fit_line, right_slope = get_fitline(frame, R_lines)

        #왼 or 오른쪽 선 기울기 필터링 및 기울기가 갑자기 바뀌는것 제외
        if abs(left_slope)<0.6 or left_slope>0:
            left_fit_line = None
        if abs(right_slope)<0.6 or right_slope<0:
            right_fit_line = None

        #차선 이탈 감지(차선 검출 안 될 시)와 이 때 이전 차선으로 유지
        if left_fit_line is None or right_fit_line is None:
            left_fit_line = prior_left_line
            right_fit_line = prior_right_line
            i = i + 1
            print(i)
            if i == 6:
                print("차선변경")
                i = 0
                return "change", prior_left_line, prior_right_line,left_fit_line, right_fit_line, i
        else:
            i = 0
            prior_left_line = left_fit_line
            prior_right_line = right_fit_line

    if prior_right_line == None or prior_left_line == None:
        return False,0,0,0,0,0

    # 대표선 그리기
    draw_fit_line(temp, left_fit_line)
    draw_fit_line(temp, right_fit_line)
    result = weighted_img(temp, frame)# 원본 이미지에 검출된 선 overlap
    cv2.imshow('result2', result)  # 결과 이미지 출력

    return "pass", prior_left_line, prior_right_line, left_fit_line, right_fit_line, i
    # result = weighted_img(temp, frame)# 원본 이미지에 검출된 선 overlap

