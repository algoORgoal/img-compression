# import numpy as np
# import cv2 as cv

import numpy
import cv2
from numpy import linalg as LA



# img_chopped_off = img[start_row: start_row+ 512, start_col:start_col+512]
# print(img_chopped_off.shape)

# display an image
# cv2.imshow("Kim", img_chopped_off)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def haar(n):
    if n == 1:
        return numpy.array([[1]])
    half = n // 2
    H = haar(half)
    I = numpy.identity(half, dtype = numpy.int16)
    left = numpy.kron(H, numpy.array([[1], [1]]))
    right = numpy.kron(I, numpy.array([[1], [-1]]))
    merged = numpy.concatenate((left, right), axis=1)
    norm = LA.norm(merged, axis=0)
    merged = numpy.divide(merged, norm)
    return merged
    
def chop(img, start=500, length = 9):
    return img[start: start + 2 ** length, start: start + 2 ** length]

def compress(img, k = 1):
    upper_left = img[: 2 ** k, : 2 ** k]
    return upper_left
    
def scale_up(upper_left, shape):
    compressed = numpy.zeros(shape, numpy.int64)
    compressed[: upper_left.shape[0], : upper_left.shape[1]] = upper_left
    return compressed


def main():
    n = int(input("type a natural number and press enter: "))
    img = cv2.imread(r"C:\Users\LG\Desktop\Projects\img-compression\images\img3.jpg", cv2.IMREAD_GRAYSCALE)
    img_chopped = chop(img)
    print(img_chopped.shape)
    print(img_chopped)
    shape = img_chopped.shape
    H = haar(2**9)
    H_t = H.transpose()
    transformed = numpy.matmul(numpy.matmul(H_t, img_chopped), H)
    upper_left = compress(transformed, 2 ** 9)
    upper_left_with_zeros = scale_up(upper_left, shape)
    result = numpy.matmul(numpy.matmul(H, upper_left_with_zeros), H_t)
    print(result.shape)
    print(result)
    # int_result = result.astype(numpy.int16)
    # print(int_result.shape)
    # print(int_result)


    cv2.imshow("original", img_chopped)
    cv2.imshow("compressed", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



main()



# print(haar(2))
# print(haar(4))

# print(numpy.matmul(numpy.matmul(numpy.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 0, 0], [0, 0, 1, -1]]), numpy.array([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])), numpy.array([[1, 1, 1, 0], [1, 1, -1, 0], [1, -1, 0, 1], [1, -1, 0, -1]])))
# print(numpy.matmul(numpy.matmul(numpy.array([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]) ,numpy.array([[24,0,0,0], [-8,0,0,0], [0,0,0,0],[0,0,0,0]])), numpy.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 0, 0], [0, 0, 1, -1]])))
    
    # img_compressed = compress(img, 1)
    # H = haar(n)
    # H_t = H.transpose()
    
    # cv2.imshow("Kim", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 이미지 부부분 줄였다가 늘이는 함수 구현하기

# main()
# print(haar(4).transpose())
# print(haar(4))
# print(haar(2))
# print(haar(4))
# print(haar(8))
# print(haar(16))
# print(haar(32))

# main()




# px = img[100, 100]
# print(px)
# print(img.shape)

# px = img[340, 200]
# B = img.item(3400, 200, 0)
# G = img.item(3400, 200, 1)
# R = img.item(3400, 200, 2)
# BGR = [B,G,R]
# print(BGR)




# px = img[340, 200]
# print(px)
# print(img.shape)
# print(img.size)
# print(img.dtype)