"""
-*- coding: utf-8 -*-
@Time    : 2022/4/19 15:08
@Emaik   :yang18319402515@163.com
@Author  : yangHuaLin
@FileName: 眉毛数据采集预测类.py
@Software: PyCharm
"""
# coding=utf-8
import numpy as np
import os
import dlib
import cv2
from PIL import ImageDraw, ImageFont, Image  # 自带图像处理包，和opencv一样
from scipy.spatial import distance  # distance距离计算工具
from imutils import face_utils  # imutils包可以很简洁的调用opencv接口
from sklearn import svm
import joblib
import uiautomator2 as u2


# 眨眼检测论文 http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf?spm=5176.100239.blogcont336184.8.b7697a07zKT7r&file=05.pdf
# 参考文章1 https://developer.aliyun.com/article/336184
# 参考文章2 https://blog.csdn.net/hongbin_xu/article/details/79033116
class Raise():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()  # 人脸检测器，检测出人脸
        self.predictor = dlib.shape_predictor(
            r"shape_predictor_68_face_landmarks.dat")  # 人脸68个关键点提取器
        cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)  # 窗口,自动调整窗口宽度
        self.cap = cv2.VideoCapture(0)  # 打开摄像头

        self.VECTOR_SIZE = 3  # 特征向量维度，这里定义3个维度，可以理解为xyz坐标向量，也是连续3帧图像的纵横比存为一个特征向量
        self.EYE_AR_CONSEC_FRAMES = 3  # 帧数阈值，当相机捕捉连续帧数是一定发生眉毛运动

    # 提取特征向量
    def queue_in(self, queue, data):
        """
        :param queue:
        :param data:
        :return:
        """
        ret = None
        if len(queue) >= self.VECTOR_SIZE:
            ret = queue.pop(0)
        queue.append(data)
        return ret, queue

    # 计算眉毛和下眼皮横纵比，可以用这个比值来判断眉毛是否在动，通过横纵比达到一定的阈值出发adb工具达到控制手机模拟滑动的效果
    def eye_aspect_ratio(self, eye):
        """
        :param eye: 携带眉毛关键点和下眼皮关键点的坐标数组,一边眼睛
        :return: 横纵比
        欧几里得范数（Euclidean norm） ==欧式长度 =L2 范数 ==L2距离 求模
        """
        # print(eye)
        A = distance.euclidean(eye[4], eye[6])  # 计算欧式距离，根据眨眼论文公式拓展到眉毛
        B = distance.euclidean(eye[3], eye[7])
        C = distance.euclidean(eye[2], eye[8])
        D = distance.euclidean(eye[1], eye[9])
        E = distance.euclidean(eye[0], eye[5])
        ear = (A + B + C + D) / (2.0 * E)
        return ear

    # 函数实现图像上写上中文，cv2格式传入，图片中文写文本，输出cv2类型
    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=20):
        # 判断是否为opencv图片类型
        if (isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转ImageDraw可处理类型
        draw = ImageDraw.Draw(img)  # 图片画板
        fontText = ImageFont.truetype('simsun.ttc', textSize, encoding="utf-8")  # 字体
        draw.text((left, top), text, textColor, font=fontText)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # 上挑眉毛数据采集
    def upEyebrow(self):
        print('上挑眉毛数据采集')
        print('按 s 开始上挑眉毛数据采集')
        print('按 e 暂停上挑眉毛数据采集')
        print('按 q 退出上挑眉毛数据采集')
        flag = 0  # 定义一个flag确保实在摄像头数据是处在采集状态
        txt = open('test_train_up.txt', 'wb')  # 打开一个文件，存储眉毛上挑的数据
        data_counter = 0  # 数据条数计数器
        ear_vector = []
        while (1):
            ret, frame = self.cap.read()
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
            if flag == 1:  # 采集状态，开始预测进入人脸检测，关键点提取
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度图
                rects = self.detector(gray, 0)  # 人脸检测，可迭代对象，存放多个人脸
                for rect in rects:  # 迭代每一张人脸进行关键点检测
                    shape = self.predictor(gray, rect)
                    # points = shape.parts()# 68个关键点坐标x,y
                    # 数据类型转换，坐标转换为数组ndarray
                    points = face_utils.shape_to_np(shape)
                    # 眉毛关键点和眼睛下眼皮关键点
                    leftEye = np.array(
                        [points[27], points[39], points[40], points[41], points[36], points[17], points[18], points[19],
                         points[20], points[21]])
                    rightEye = np.array(
                        [points[27], points[42], points[47], points[46], points[45], points[26], points[25], points[24],
                         points[23], points[22]])
                    leftEAR = self.eye_aspect_ratio(leftEye)  # 左眼眼睛纵横比
                    rightEAR = self.eye_aspect_ratio(rightEye)  # 右眼眼睛纵横比
                    ear = (leftEAR + rightEAR) / 2.0  # 取左右眼横纵比的均值

                    leftEyeHull = cv2.convexHull(leftEye)  # 传入眼睛下眼皮和眉毛的坐标，计算凸包及更多轮廓特征
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # 轮廓绘制
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    ret, ear_vector = self.queue_in(ear_vector, ear)  # 提取指定维度的特征向量
                    if (len(ear_vector) == self.VECTOR_SIZE):  # 特征向量是设定维度进行写进文件里面
                        txt.write(bytes(str(ear_vector), encoding="utf-8"))
                        txt.write(bytes("\n", encoding="utf-8"))
                        data_counter += 1
                        print(data_counter)
                    frame = self.cv2ImgAddText(frame, "纵横比:{:.2f}".format(ear), 10, 30, textColor=(255, 128, 0),
                                               textSize=40)
            cv2.imshow("frame", frame)
        txt.close()  # 关闭文件

    #  正常情况
    def eyeNormal(self):
        print('-' * 40)
        print('正常情况数据采集')
        print('按 s 开始正常情况数据采集')
        print('按 e 暂停正常情况数据采集')
        print('按 q 退出正常情况数据采集')
        flag = 0
        txt = open('test_train_normal.txt', 'wb')
        data_counter = 0
        ear_vector = []
        while (1):
            ret, frame = self.cap.read()
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
                    points = face_utils.shape_to_np(shape)
                    # points = shape.parts()
                    leftEye = np.array(
                        [points[27], points[39], points[40], points[41], points[36], points[17], points[18], points[19],
                         points[20], points[21]])
                    rightEye = np.array(
                        [points[27], points[42], points[47], points[46], points[45], points[26], points[25], points[24],
                         points[23], points[22]])
                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)

                    ear = (leftEAR + rightEAR) / 2.0

                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    ret, ear_vector = self.queue_in(ear_vector, ear)
                    if (len(ear_vector) == self.VECTOR_SIZE):
                        txt.write(bytes(str(ear_vector), encoding="utf-8"))
                        txt.write(bytes("\n", encoding="utf-8"))
                        data_counter += 1
                        print(data_counter)
                    frame = self.cv2ImgAddText(frame, "纵横比:{:.2f}".format(ear), 10, 30, textColor=(255, 128, 0),
                                               textSize=40)
            cv2.imshow("frame", frame)
        txt.close()

    # 下皱数据采集
    def downEyebrow(self):
        print('-' * 40)
        print('下皱眉毛数据采集')
        print('按 s 开始下皱眉毛数据采集')
        print('按 e 暂停下皱眉毛数据采集')
        print('按 q 退出下皱眉毛数据采集')
        flag = 0
        txt = open('test_train_down.txt', 'wb')
        data_counter = 0
        ear_vector = []
        while (1):
            ret, frame = self.cap.read()
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
                    points = face_utils.shape_to_np(shape)
                    # points = shape.parts()

                    leftEye = np.array(
                        [points[27], points[39], points[40], points[41], points[36], points[17], points[18], points[19],
                         points[20], points[21]])
                    rightEye = np.array(
                        [points[27], points[42], points[47], points[46], points[45], points[26], points[25], points[24],
                         points[23], points[22]])
                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    ret, ear_vector = self.queue_in(ear_vector, ear)
                    if (len(ear_vector) == self.VECTOR_SIZE):
                        txt.write(bytes(str(ear_vector), encoding="utf-8"))
                        txt.write(bytes("\n", encoding="utf-8"))
                        data_counter += 1
                        print(data_counter)
                    frame = self.cv2ImgAddText(frame, "纵横比:{:.2f}".format(ear), 10, 30, textColor=(255, 128, 0),
                                               textSize=40)
            cv2.imshow("frame", frame)
        txt.close()

    # 训练SVM眨眼检测
    def eyeFit(self):
        self.cap.release()
        cv2.destroyAllWindows()
        train = []
        labels = []
        train_up_txt = open("data/train_up.txt", "r")
        train_normal_txt = open("data/train_normal.txt", "r")
        train_down_txt = open("data/train_down.txt", "r")

        print('Reading train_up.txt...')
        for txt_str in train_up_txt.readlines():
            temp = []
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
            labels.append(1)  # 标签

        print('Reading train_normal.txt...')
        # temp = []
        for txt_str in train_normal_txt.readlines():
            temp = []
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
            labels.append(0)  # 标签

        print('Reading train_down.txt...')
        # temp = []
        for txt_str in train_down_txt.readlines():
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
            labels.append(-1)  # 标签

        for i in range(len(labels)):
            print("{0} --> {1}".format(train[i], labels[i]))

        train_up_txt.close()
        train_down_txt.close()
        train_normal_txt.close()
        # 支持向量机分类器惩罚系数，线性核，核函数系数 ，ovo一对一分类
        clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
        clf.fit(train, labels)
        joblib.dump(clf, "sysUpDown_svm.m")  # 保存模型
        print('predicting [[0.88, 0.85, 0.90]]')
        res = clf.predict([[0.88, 0.85, 0.90]])
        print(res)

        print('predicting [[0.60, 0.62, 0.60]]')
        res = clf.predict([[0.60, 0.62, 0.60]])
        print(res)
        return train, labels

    # 预测
    def pre(self):
        # 导入模型
        clf = joblib.load("sysUpDown_svm.m")
        # 阈值控制
        EYE_AR_UP = 0.9876  # 上挑眉毛EAR阈值，大于阈值触发手机上滑动下一条抖音
        EYE_AR_DOWN = 0.6434  # 下皱眉毛EAR阈值，小于阈值触发手机下滑动上一条抖音
        EYE_AR_CONSEC_FRAMES = 5  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作

        flag = True  # 摄像头帧很快，防止连续帧会多次触发手机，用一个全局变量把在一次触发手机滑动之后，更新为false
        up_frame_counter = 0  # 上挑眉毛连续帧数记录计数器帧数
        down_frame_counter = 0  # 下皱眉毛连续帧数记录计数器帧数
        up_counter = 0  # 上挑眉毛次数
        down_counter = 0  # 下皱眉毛次数
        ear_vector = []
        # device = u2.connect("10.50.120.171:9999")  # 连接手机的IP和端口，要在局域网内
        device = u2.connect("127.0.0.1:21503")  # 连接模拟器
        while (1):
            ret, img = self.cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                points = face_utils.shape_to_np(shape)
                # points = shape.parts()
                leftEye = np.array(
                    [points[27], points[39], points[40], points[41], points[36], points[17], points[18], points[19],
                     points[20], points[21]])
                rightEye = np.array(
                    [points[27], points[42], points[47], points[46], points[45], points[26], points[25], points[24],
                     points[23], points[22]])
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

                ret, ear_vector = self.queue_in(ear_vector, ear)  # 预测不一定只是一个人脸，所以定义一个

                if (len(ear_vector) == self.VECTOR_SIZE):
                    input_vector = []
                    input_vector.append(ear_vector)
                    res = clf.predict(input_vector)
                    # print(res)
                    if res == 1:  # 预测结果为1,眉毛上挑，看有多少连续帧
                        # 两个阈值满足条件才能触发手机滑动，达到帧数阈值识别为确定你在挑眉毛，达到触发纵横比阈值才确定你想要滑动手机
                        if up_frame_counter >= EYE_AR_CONSEC_FRAMES and ear >= EYE_AR_UP:  # 确定为超过
                            up_counter += 1
                            if flag:
                                print("上滑")
                                flag = False  # 更新flag，防止连续触发滑动
                                # device.shell("input swipe 540 1665 540 722") # 真机
                                device.shell("input swipe 360 910 360 480")  # 触发滑动模拟器
                        up_frame_counter += 1  # 连续预测结果帧数叠加
                    elif res == -1:  # 预测结果是2，眉毛下皱
                        if down_frame_counter >= EYE_AR_CONSEC_FRAMES and ear <= EYE_AR_DOWN:  # 确定为超过
                            down_counter += 1
                            if flag:
                                print("下滑")
                                flag = False
                                # device.shell("input swipe 540 722 540 1665")
                                device.shell("input swipe 360 480 360 910")
                        down_frame_counter += 1  # 连续预测结果帧数叠加
                    else:  # 否则正常，无操作
                        up_frame_counter = 0  # 循环一次预测结果非0初始化一次
                        down_frame_counter = 0  # 循环一次预测结果非0初始化一次
                        flag = True
                # 眉毛运动计数器显示不准确
                img = self.cv2ImgAddText(img, "上挑眉次数:{0}".format(up_counter), 10, 30, textColor=(255, 0, 0),
                                         textSize=30)
                img = self.cv2ImgAddText(img, "下皱眉次数:{0}".format(down_counter), 10, 70, textColor=(255, 0, 0),
                                         textSize=30)
                img = self.cv2ImgAddText(img, "纵横比:{:.4f}".format(ear), 340, 30, textColor=(255, 0, 0),
                                         textSize=30)
                for i in range(17, 68):
                    cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
            # print(round(ear, 2))
            cv2.imshow("frame", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # eye = BlinkEye()
    # eye.upEyebrow()
    # eye.downEyebrow()
    r = Raise()
    # r.upEyebrow()
    # r.eyeNormal()
    # r.downEyebrow()
    # train, labels = r.eyeFit()
    r.pre()
    # print(train)
    # print(labels)
