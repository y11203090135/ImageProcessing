# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:20:52 2018

@author: mie
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb

#pdb.set_trace()
def corner_detect(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (1280, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cornerHarris函数图像格式为 float32 ，因此需要将图像转换 float32 类型
    gray = np.float32(gray)
    # cornerHarris参数：
    # src - 数据类型为 float32 的输入图像。
    # blockSize - 角点检测中要考虑的领域大小。
    # ksize - Sobel 求导中使用的窗口大小
    # k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06].
    dst = cv2.cornerHarris(src=gray, blockSize=9, ksize=23, k=0.04)
    # 变量a的阈值为0.01 * dst.max()，如果dst的图像值大于阈值，那么该图像的像素点设为True，否则为False
    # 将图片每个像素点根据变量a的True和False进行赋值处理，赋值处理是将图像角点勾画出来
    a = dst>0.01 * dst.max()
    img[a] = [0, 0, 255]
    # 显示图像
    while (True):
        cv2.imshow('corners', img)
        if cv2.waitKey(0) & 0xff == ord("q"):
            break
        cv2.destroyAllWindows()
        
def sift_detect(imgpath):
    # 读取图片并灰度处理
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建SIFT对象
    sift = cv2.xfeatures2d.SIFT_create()
    # 将图片进行SURF计算，并找出角点keypoints，keypoints是检测关键点
    # descriptor是描述符，这是图像一种表示方式，可以比较两个图像的关键点描述符，可作为特征匹配的一种方法。
    keypoints, descriptor = sift.detectAndCompute(gray, None)

    # cv2.drawKeypoints() 函数主要包含五个参数：
    # image: 原始图片
    # keypoints：从原图中获得的关键点，这也是画图时所用到的数据
    # outputimage：输出
    # color：颜色设置，通过修改（b,g,r）的值,更改画笔的颜色，b=蓝色，g=绿色，r=红色。
    # flags：绘图功能的标识设置，标识如下：
    # cv2.DRAW_MATCHES_FLAGS_DEFAULT  默认值
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    # cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints = keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT, color = (51, 163, 236))

    # 显示图片
    cv2.imshow('sift_keypoints', img)
    while (True):
      if cv2.waitKey(120) & 0xff == ord("q"):
        break
    cv2.destroyAllWindows()
    
def surf_detect(imgpath):
    # 读取图片并灰度处理
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建SIFT对象
    sift = cv2.xfeatures2d.SURF_create(float(4000))
    # 将图片进行SURF计算，并找出角点keypoints，keypoints是检测关键点
    # descriptor是描述符，这是图像一种表示方式，可以比较两个图像的关键点描述符，可作为特征匹配的一种方法。
    keypoints, descriptor = sift.detectAndCompute(gray, None)

    # cv2.drawKeypoints() 函数主要包含五个参数：
    # image: 原始图片
    # keypoints：从原图中获得的关键点，这也是画图时所用到的数据
    # outputimage：输出
    # color：颜色设置，通过修改（b,g,r）的值,更改画笔的颜色，b=蓝色，g=绿色，r=红色。
    # flags：绘图功能的标识设置，标识如下：
    # cv2.DRAW_MATCHES_FLAGS_DEFAULT  默认值
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    # cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints = keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT, color = (51, 163, 236))

    # 显示图片
    cv2.imshow('sift_keypoints', img)
    while (True):
      if cv2.waitKey(120) & 0xff == ord("q"):
        break
    cv2.destroyAllWindows()
    
def orb_detect(imgpath1, imgpath2):
    # 读取图片内容
    #pdb.set_trace()
    img1 = cv2.imread(imgpath1,0)
    img2 = cv2.imread(imgpath2,0)

    # 使用ORB特征检测器和描述符，计算关键点和描述符
    orb = cv2.ORB_create()
    #orb = cv2.xfeatures2d.SURF_create(float(4000))
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)


    # 暴力匹配BFMatcher，遍历描述符，确定描述符是否匹配，然后计算匹配距离并排序
    # BFMatcher函数参数：
    # normType：NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2。
    # NORM_L1和NORM_L2是SIFT和SURF描述符的优先选择，NORM_HAMMING和NORM_HAMMING2是用于ORB算法
    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    #bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
    
    # 由于匹配顺序是：matches = bf.match(des1,des2)，先des1后des2。
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    # matches是DMatch对象，具有以下属性：
    # DMatch.distance - 描述符之间的距离。 越低越好。
    # DMatch.trainIdx - 训练描述符中描述符的索引
    # DMatch.queryIdx - 查询描述符中描述符的索引
    # DMatch.imgIdx - 训练图像的索引。

    # 因此，kp1的索引由DMatch对象属性为queryIdx决定，kp2的索引由DMatch对象属性为trainIdx决定

    # 获取aa.jpg的关键点位置
    x,y = kp1[matches[0].queryIdx].pt
    cv2.rectangle(img1, (int(x),int(y)), (int(x) + 5, int(y) + 5), (0, 255, 0), 2)
    cv2.imshow('a', img1)

    # 获取bb.png的关键点位置
    x1,y1 = kp2[matches[0].trainIdx].pt
    cv2.rectangle(img2, (int(x1),int(y1)), (int(x1) + 5, int(y1) + 5), (0, 255, 0), 2)
    cv2.imshow('b', img2)

    # 使用plt将两个图像的第一个匹配结果显示出来
    img3 = cv2.drawMatches(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2, matches1to2=matches[:1], outImg=img2, flags=2)
    plt.imshow(img3),plt.show()

    
    # 使用plt将两个图像的匹配结果显示出来
    #img3 = cv2.drawMatches(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2, matches1to2=matches, outImg=img2, flags=2)
    #plt.imshow(img3),plt.show()

def knn_match(imgpath1, imgpath2):
# 读取图片内容
    img1 = cv2.imread(imgpath1,0)
    img2 = cv2.imread(imgpath2,0)

    # 使用ORB特征检测器和描述符，计算关键点和描述符
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # 暴力匹配BFMatcher，遍历描述符，确定描述符是否匹配，然后计算匹配距离并排序
    # BFMatcher函数参数：
    # normType：NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2。
    # NORM_L1和NORM_L2是SIFT和SURF描述符的优先选择，NORM_HAMMING和NORM_HAMMING2是用于ORB算法
    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    # knnMatch 函数参数k是返回符合匹配的个数，暴力匹配match只返回最佳匹配结果。
    matches = bf.knnMatch(des1,des2,k=1)

    # 使用plt将两个图像的第一个匹配结果显示出来
    # 若使用knnMatch进行匹配，则需要使用drawMatchesKnn函数将结果显示
    img3 = cv2.drawMatchesKnn(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2, matches1to2=matches, outImg=img2, flags=2)
    plt.imshow(img3),plt.show()
    
def flann_match(imgpath1, imgpath2):
    queryImage = cv2.imread(imgpath1,0)
    trainingImage = cv2.imread(imgpath2,0)

    # 只使用SIFT 或 SURF 检测角点
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.xfeatures2d.SURF_create(float(4000))
    kp1, des1 = sift.detectAndCompute(queryImage,None)
    kp2, des2 = sift.detectAndCompute(trainingImage,None)

    # 设置FLANN匹配器参数
    # algorithm设置可参考https://docs.opencv.org/3.1.0/dc/d8c/namespacecvflann.html
    indexParams = dict(algorithm=0, trees=5)
    searchParams = dict(checks=50)
    # 定义FLANN匹配器
    flann = cv2.FlannBasedMatcher(indexParams,searchParams)
    # 使用 KNN 算法实现匹配
    matches = flann.knnMatch(des1,des2,k=2)

    # 根据matches生成相同长度的matchesMask列表，列表元素为[0,0]
    matchesMask = [[0,0] for i in range(len(matches))]

    # 去除错误匹配
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]

    # 将图像显示
    # matchColor是两图的匹配连接线，连接线与matchesMask相关
    # singlePointColor是勾画关键点
    drawParams = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
    resultImage = cv2.drawMatchesKnn(queryImage,kp1,trainingImage,kp2,matches,None,**drawParams)
    plt.imshow(resultImage,),plt.show()


#surf_detect('12.jpg')
flann_match('11.jpg', '12.jpg')
