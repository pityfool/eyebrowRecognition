"""
-*- coding: utf-8 -*-
@Time    : 2022/4/19 15:44
@Emaik   :yang18319402515@163.com
@Author  : yangHuaLin
@FileName: 眨眼预测.py
@Software: PyCharm
"""
# coding=utf-8
import numpy as np
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
from sklearn import svm
import joblib

VECTOR_SIZE = 3


# 眨眼次数检测
class BlinkEye():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            r"shape_predictor_68_face_landmarks.dat")

        # 对应特征点的序号
        self.RIGHT_EYE_START = 37 - 1
        self.RIGHT_EYE_END = 42 - 1
        self.LEFT_EYE_START = 43 - 1
        self.LEFT_EYE_END = 48 - 1
        self.VECTOR_SIZE = 3

        self.EYE_AR_THRESH = 0.3  # EAR阈值
        self.EYE_AR_CONSEC_FRAMES = 3  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作

        # 对应特征点的序号
        self.RIGHT_EYE_START = 37 - 1
        self.RIGHT_EYE_END = 42 - 1
        self.LEFT_EYE_START = 43 - 1

        self.LEFT_EYE_END = 48 - 1

        self.frame_counter = 0
        self.blink_counter = 0
        self.ear_vector = []

    def queue_in(self, queue, data):
        ret = None
        if len(queue) >= self.VECTOR_SIZE:
            ret = queue.pop(0)
        queue.append(data)
        return ret, queue

    # cv2传入，图片中文写文本，输出cv2类型
    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=20):
        # 判断是否为opencv图片类型
        if (isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype('simsun.ttc', textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontText)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def eye_aspect_ratio(self, eye):
        # print(eye)
        A = distance.euclidean(eye[1], eye[5])  # 计算欧式距离
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    # 睁眼数据采集
    def openEys(self):
        cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
        cap = cv2.VideoCapture(0)
        print('睁眼数据采集')
        print('按 s 开始数据采集')
        print('按 e 暂停数据采集')
        print('按 q 退出数据采集')
        flag = 0
        txt = open('data/train_open.txt', 'wb')
        data_counter = 0
        ear_vector = []
        while (1):
            ret, frame = cap.read()
            key = cv2.waitKey(1)
            if key & 0xFF == ord("s"):
                print('开始数据采集')
                flag = 1
            elif key & 0xFF == ord("e"):
                print('暂停数据采集')
                flag = 0
            elif key & 0xFF == ord("q"):
                print('退出数据采集')
                break

            if flag == 1:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    # convert the facial landmark (x, y)-coordinates to a NumPy array
                    points = face_utils.shape_to_np(shape)
                    # points = shape.parts()
                    leftEye = points[self.LEFT_EYE_START:self.LEFT_EYE_END + 1]  # 左眼睛切片
                    rightEye = points[self.RIGHT_EYE_START:self.RIGHT_EYE_END + 1]  # 右眼睛切片
                    leftEAR = self.eye_aspect_ratio(leftEye)  # 左眼眼睛纵横比
                    rightEAR = self.eye_aspect_ratio(rightEye)  # 右眼眼睛纵横比
                    # print('leftEAR = {0}'.format(leftEAR))
                    # print('rightEAR = {0}'.format(rightEAR))

                    ear = (leftEAR + rightEAR) / 2.0

                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    ret, ear_vector = self.queue_in(ear_vector, ear)
                    if (len(ear_vector) == self.VECTOR_SIZE):
                        # print(ear_vector)
                        # inpuself.t_vector = []
                        # input_vector.append(ear_vector)

                        txt.write(bytes(str(ear_vector), encoding="utf-8"))
                        txt.write(bytes("\n", encoding="utf-8"))
                        data_counter += 1
                        print(data_counter)

                    cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                                2)

            cv2.imshow("frame", frame)
        txt.close()
        cap.release()
        cv2.destroyAllWindows()

    # 闭眼数据采集
    def closeEye(self):
        cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)

        cap = cv2.VideoCapture(0)
        print('-' * 40)
        print('闭眼眼数据采集')
        print('按 s 开始数据采集')
        print('按 e 暂停数据采集')
        print('按 q 退出数据采集')
        flag = 0
        txt = open('train_close.txt', 'wb')
        data_counter = 0
        ear_vector = []
        while (1):
            ret, frame = cap.read()
            key = cv2.waitKey(1)
            if key & 0xFF == ord("s"):
                print('开始数据采集')
                flag = 1
            elif key & 0xFF == ord("e"):
                print('暂停数据采集')
                flag = 0
            elif key & 0xFF == ord("q"):
                print('退出数据采集')
                break

            if flag == 1:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    points = face_utils.shape_to_np(
                        shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
                    # points = shape.parts()
                    leftEye = points[self.LEFT_EYE_START:self.LEFT_EYE_END + 1]
                    rightEye = points[self.RIGHT_EYE_START:self.RIGHT_EYE_END + 1]
                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    # print('leftEAR = {0}'.format(leftEAR))
                    # print('rightEAR = {0}'.format(rightEAR))

                    ear = (leftEAR + rightEAR) / 2.0

                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    ret, ear_vector = self.queue_in(ear_vector, ear)
                    if (len(ear_vector) == self.VECTOR_SIZE):
                        # print(ear_vector)
                        # input_vector = []
                        # input_vector.append(ear_vector)

                        txt.write(bytes(str(ear_vector), encoding="utf-8"))
                        txt.write(bytes("\n", encoding="utf-8"))

                        data_counter += 1
                        print(data_counter)

                    cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                                2)

            cv2.imshow("frame", frame)
        txt.close()
        cap.release()
        cv2.destroyAllWindows()

    # 训练SVM眨眼检测
    def eyeFit(self):
        train = []
        labels = []
        train_open_txt = open("data/train_open.txt", "r")
        train_close_txt = open("train_close.txt", "r")
        print('Reading train_open.txt...')
        line_ctr = 0
        for txt_str in train_open_txt.readlines():
            temp = []
            # print(txt_str)
            datas = txt_str.strip()
            datas = datas.replace('[', '')
            datas = datas.replace(']', '')
            datas = datas.split(',')
            print(datas)
            for data in datas:
                # print(data)
                data = float(data)
                temp.append(data)
            # print(temp)
            train.append(temp)
            labels.append(0)

        print('Reading train_close.txt...')
        line_ctr = 0
        temp = []
        for txt_str in train_close_txt.readlines():
            temp = []
            # print(txt_str)
            datas = txt_str.strip()
            datas = datas.replace('[', '')
            datas = datas.replace(']', '')
            datas = datas.split(',')
            print(datas)
            for data in datas:
                # print(data)
                data = float(data)
                temp.append(data)
            # print(temp)
            train.append(temp)
            labels.append(1)

        for i in range(len(labels)):
            print("{0} --> {1}".format(train[i], labels[i]))

        train_close_txt.close()
        train_open_txt.close()
        clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
        clf.fit(train, labels)
        joblib.dump(clf, "ear_svm.m")
        print('predicting [[0.34, 0.34, 0.31]]')
        res = clf.predict([[0.34, 0.34, 0.31]])
        print(res)

        print('predicting [[0.19, 0.18, 0.18]]')
        res = clf.predict([[0.19, 0.18, 0.18]])
        print(res)
        return train, labels

    def pre(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            r"shape_predictor_68_face_landmarks.dat")
        # 导入模型
        # clf = joblib.load("ear_svm.m")
        clf = joblib.load("ear_svm.m")
        EYE_AR_CONSEC_FRAMES = 3  # 当纵横比小于阈值时，接连多少帧一定发生眨眼动作

        # 对应特征点的序号
        RIGHT_EYE_START = 37 - 1
        RIGHT_EYE_END = 42 - 1
        LEFT_EYE_START = 43 - 1
        LEFT_EYE_END = 48 - 1

        frame_counter = 0
        blink_counter = 0
        ear_vector = []
        cap = cv2.VideoCapture(0)
        while (1):
            ret, img = cap.read()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                print('-' * 20)
                shape = predictor(gray, rect)
                points = face_utils.shape_to_np(
                    shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
                # points = shape.parts()
                leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
                rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                print('leftEAR = {0}'.format(leftEAR))
                print('rightEAR = {0}'.format(rightEAR))

                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

                ret, ear_vector = self.queue_in(ear_vector, ear)
                if (len(ear_vector) == self.VECTOR_SIZE):
                    print(ear_vector)
                    input_vector = []
                    input_vector.append(ear_vector)
                    res = clf.predict(input_vector)
                    print(res)

                    if res == 1:
                        frame_counter += 1
                    else:
                        if frame_counter >= EYE_AR_CONSEC_FRAMES:
                            blink_counter += 1
                        frame_counter = 0
                cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue


def eye_aspect_ratio(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    r"E:\workspace\python_work\code\hengdaProject\scienceApp\deepLearningModels\shape_predictor_68_face_landmarks.dat")
# 导入模型
clf = joblib.load("ear_svm.m")
# clf = joblib.load("sysUpDown_svm.m")

EYE_AR_THRESH = 0.3  # EAR阈值
EYE_AR_CONSEC_FRAMES = 3  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

frame_counter = 0
blink_counter = 0
ear_vector = []
cap = cv2.VideoCapture(0)
while (1):
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        print('-' * 20)
        shape = predictor(gray, rect)
        points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
        # points = shape.parts()
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        print('leftEAR = {0}'.format(leftEAR))
        print('rightEAR = {0}'.format(rightEAR))

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

        ret, ear_vector = queue_in(ear_vector, ear)
        if (len(ear_vector) == VECTOR_SIZE):
            print(ear_vector)
            input_vector = []
            input_vector.append(ear_vector)
            res = clf.predict(input_vector)
            print(res)

            if res == 1:
                frame_counter += 1
            else:
                if frame_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_counter += 1
                frame_counter = 0

        cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
