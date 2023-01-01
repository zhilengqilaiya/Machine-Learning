# 导包
import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import fabs, sin, cos, radians, log10, sqrt
import scipy.stats as stats

init_img = cv2.imread(r'C:\Users\CZQ\Desktop\Chapter2_1.pgm')
init_angle = 15
init_height, init_width = init_img.shape[:2]


def rotated(img, angle, method):
    # 获取宽高,旋转中心
    print(img.shape)
    height, width = img.shape[:2]
    x = width / 2
    y = height / 2

    # 获取初始旋转矩阵
    # getRotationMatrix2D(旋转中心,旋转角度,缩放比例),变换矩阵,填入到wrapAffine仿射变换的参数)
    # 旋转中选需要采用原图像的中心
    rotate_matrix = cv2.getRotationMatrix2D((x, y), angle, 1)
    print(rotate_matrix)

    # 计算旋转后的显示范围,加1防止误差丢点
    rotated_height = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle)))) + 1
    rotated_width = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle)))) + 1

    # 修改旋转矩阵
    """
    [cos -sin (1-cos)*x + sin*y
    sin cos  (1-cos)*y - sin*x ]
    """
    rotate_matrix[0, 2] += (rotated_width - width) / 2
    rotate_matrix[1, 2] += (rotated_height - height) / 2

    # 旋转图像
    rotated_img = cv2.warpAffine(img, rotate_matrix, (rotated_width, rotated_height), None, method)
    return rotated_img


def recovered(img, angle, method):
    # 获取宽高,旋转中心
    print(img.shape)
    height, width = img.shape[:2]
    x = width / 2
    y = height / 2

    # 获取初始旋转矩阵
    # getRotationMatrix2D(旋转中心,旋转角度,缩放比例),变换矩阵,填入到wrapAffine仿射变换的参数)
    # 旋转中选需要采用原图像的中心
    rotate_matrix = cv2.getRotationMatrix2D((x, y), angle, 1)
    print(rotate_matrix)

    # 计算旋转后的显示范围,目标区域不会因为恢复而丢点
    rotated_height = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    rotated_width = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    # 修改旋转矩阵
    rotate_matrix[0, 2] += (rotated_width - width) / 2
    rotate_matrix[1, 2] += (rotated_height - height) / 2

    # 旋转图像
    rotated_img = cv2.warpAffine(img, rotate_matrix, (rotated_width, rotated_height), None, method)

    # 计算图像起始坐标
    start_width = (len(rotated_img[0]) - len(init_img[0])) >> 1
    start_height = (len(rotated_img) - len(init_img)) >> 1

    print(start_width, start_height, width, height, init_width, init_height)
    print(start_width, start_height, len(rotated_img[0]), len(rotated_img), len(init_img[0]), len(init_img))

    # img[height,width]
    return rotated_img[start_height:start_height + init_height, start_width:start_width + init_width]
    # return rotated_img


# 最近邻 旋转十五度
rotated_img_nearest = rotated(init_img, init_angle, cv2.INTER_NEAREST)
plt.title("nearest_rotated")
plt.imshow(rotated_img_nearest)