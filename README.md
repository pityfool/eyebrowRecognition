# 基于机器学习算法、人脸识别算法、计算机视觉开发的眉毛识别控制抖音刷短视频案例

###### 彻底理解原理所需知识：

​	1.高等数学、线性代数

​	2.python编程基础

​	3.机器学习算法基础

​	4.opencv计算机视觉开发基础

​	5.linux常用操作命令

​	6.通信原理

程序设置架构原理图

![架构设置](.\架构设置.png)

###### 人脸68关键点索引

![68关键点索引](.\68关键点索引.png)

###### 第三方依赖环境

```shell
numpy dlib opencv-python scipy imutils sklearn joblib uiautomator2  
```

###### 准备条件

先打开手机的开发者工具，启用usb调试功能，再用adb工具对手机进行连接

```shell
adb.exe usb
adb.exe tcpip 9999
adb.exe connect 你手机的IP:9999
```

连接成功后才能使用代码连接

###### 纵横比计算公式

$$ {左眼纵横比}
LeftEAR=(||index18-index36||+ ||index10-index41||+ ||index20-index40||+ ||index21-index39||)/(2||index17-index27||)
$$

$$ {右眼纵横比}
RightEAR=(||index25-index45||+ ||index24-index46||+ ||index23-index47||+ ||index22-index42||)/(2||index26-index27||)
$$

###### 个人觉得该算法还有优化的思路

通过摇头或人在动时识别就不准确，或许可以根据人脸计算正常情况下的比例计算与识别是的比例进行比较在进行矫正或者再根据比例乘改阈值达到触发模拟滑动阈值也跟着人脸变换而改变。

###### 拓展

继续根据该算法添加新的功能，例如双眨眼实现双击点赞功能、张开嘴巴打开评论区，再次张开嘴巴关闭评论区等等功能。

###### 参考文献

###### 眨眼检测论文 http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf?spm=5176.100239.blogcont336184.8.b7697a07zKT7r&file=05.pdf
###### 参考文章1 https://developer.aliyun.com/article/336184
###### 参考文章2 https://blog.csdn.net/hongbin_xu/article/details/79033116

