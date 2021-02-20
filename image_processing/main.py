import cv2
import numpy as np
from utils import make_datapath_list

def main():
    ok_list = make_datapath_list('train', 'ok') + make_datapath_list('val', 'ok')
    v5_list = make_datapath_list('train', 'level5')

    min_val = 60
    max_val = 60
    mask_img = cv2.imread('../dataset/mask.jpg', cv2.IMREAD_GRAYSCALE)

    for i in range(len(v5_list)):
        print(ok_list[i])
        print(v5_list[i])
        gray_img = cv2.imread(v5_list[i], cv2.IMREAD_GRAYSCALE)
        gray_ok = cv2.imread(ok_list[i], cv2.IMREAD_GRAYSCALE)

        gray_img[mask_img == 0] = 0
        gray_ok[mask_img == 0] = 0

        ret2, otsu_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
        ret2, otsu_ok = cv2.threshold(gray_ok, 0, 255, cv2.THRESH_OTSU)

        canny_img = cv2.Canny(gray_img, min_val, max_val)
        canny_ok = cv2.Canny(gray_ok, min_val, max_val)
        otsu_canny_img = cv2.Canny(otsu_img, min_val, max_val)
        otsu_canny_ok = cv2.Canny(otsu_ok, min_val, max_val)

        # cv2.imwrite('./processed_images/lv5/lv5_{}.jpg'.format(i), canny_img)
        # cv2.imwrite('./processed_images/ok/ok_{}.jpg'.format(i), canny_ok)
        cv2.imwrite('./processed_images/lv5/otsu_lv5_{}.jpg'.format(i), otsu_canny_img)
        cv2.imwrite('./processed_images/ok/otsu_ok_{}.jpg'.format(i), otsu_canny_ok)

        if i > 9:
            break

if __name__ == '__main__':
    main()