#! /usr/bin/env python3

import copy
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Khai báo các biến sử dụng
prediction = ''
score = 0
bgModel = None

gesture_names = {0: 'E',
                 1: 'L',
                 2: 'F',
                 3: 'V',
                 4: 'B'}

# Tải model
model = load_model('models/mymodel.h5')

# Ánh xạ để lấy ký tự 
def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score


# Xóa nền ảnh
def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# Lấy kích thước nền sẽ nhận cử chỉ
cap_region_x_begin = 0.5
cap_region_y_end = 0.8

# Các thông số lấy threshold
threshold = 60
blurValue = 41
bgSubThreshold = 50 #50
learningRate = 0

# Ngưỡng dự đoán ký tự
predThreshold= 95

isBgCaptured = 0  # Biến lưu trữ đã capture background chưa

# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)

while camera.isOpened():
    # Đọc ảnh từ webcam
    ret, frame = camera.read()
    # Làm mịn ảnh
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # Lật ảnh giống gương
    frame = cv2.flip(frame, 1)

    # Vẽ khung chữ nhật vùng detection region
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    # Nếu đã capture được nền
    if isBgCaptured == 1:
        # Tách nền
        img = remove_background(frame)

        # Lấy vùng detection
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI



        # Chuyển ảnh về dạng đen trắng
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

        cv2.imshow('original1', cv2.resize(blur, dsize=None, fx=0.5, fy=0.5))

        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('thresh', cv2.resize(thresh, dsize=None, fx=0.5, fy=0.5))

        if (np.count_nonzero(thresh)/(thresh.shape[0]*thresh.shape[0])>0.2):
            # Nếu vẽ được bàn tay
            if (thresh is not None):
                # Đưa vào mảng để predict
                target = np.stack((thresh,) * 3, axis=-1)
                target = cv2.resize(target, (224, 224))
                target = target.reshape(1, 224, 224, 3)
                prediction, score = predict_rgb_image_vgg(target)

                # Nếu probality > ngưỡng dự đoán hiện tại thì
                print(score,prediction)
                if (score>=predThreshold):
                    cv2.putText(frame, "Sign:" + prediction, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                (0, 0, 255), 10, lineType=cv2.LINE_AA)
    thresh = None

    # Xử lý bàn phìm
    k = cv2.waitKey(10)
    if k == ord('q'):  # Bấm q để thoát
        break
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

        isBgCaptured = 1
        cv2.putText(frame, "Background captured", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255), 10, lineType=cv2.LINE_AA)
        time.sleep(2)
        print('Background captured')

    elif k == ord('r'):

        bgModel = None
        isBgCaptured = 0
        cv2.putText(frame, "Background reset", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255),10,lineType=cv2.LINE_AA)
        print('Background reset')
        time.sleep(1)


    cv2.imshow('original', cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))


cv2.destroyAllWindows()
camera.release()
