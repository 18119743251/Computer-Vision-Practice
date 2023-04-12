import numpy as np
import cv2
class image_Stitcher:
    # 开始定义拼接函数
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):#nfeatures，保留的最佳特性的数量。
        # 特征按其得分进行排序(以SIFT算法作为局部对比度进行测量)
        # 读取图像
        imageB, imageA = images #注意这里imageB对应图片1，imageA对应图片2，因为在之后的拼接时，是从A上覆盖B的
        # 通过函数detecAndDescribe来计算图片A与图片B的特征点和特征向量
        kpsA, featureA = self.detectAndDescribe(imageA)
        kpsB, featureB = self.detectAndDescribe(imageB)

        # 匹配两张图片的特征点
        M = self.matchKeypoints(kpsA, kpsB, featureA, featureB, ratio, reprojThresh) # 高斯输入层级， 如果图像分辨率较低，则可能需要减少数值
        # 没有匹配点，退出
        if not M:
            return None
        matches, H, status = M  # 对比度阈值用于过滤区域中的弱特征。阈值越大，检测器产生的特征越少
        # 将图片A进行视角变换 中间结果
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # 将图片B传入]
        print('A B R', imageA.shape, imageB.shape, result.shape) #这里先输出一下A、B、R的尺寸，看看能否拼接，确定宽度是一样的
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB #这最后一步比较就比较简单了，从以西代码可以看，已知举证 H，就可
        #以对图片 A 的位置进行相对变换，想象一下一个长是 A 和 B 两个图片长的画
        #布，现在通过 H 把 A 放在合适的位置（刚好可以重叠的区域的边界），那么
        #在把图像 B 也放在这个画布原本图像重叠的边界，如此图像拼接是不是就完成了。
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return result, vis
        # self.cv_show('result', result)
        # 检测是否需要显示图片匹配
        return result

    def detectAndDescribe(self, image):
        # 首先转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # featureA和featureB 表示两个特征向量集；K 表示按knn匹配规则输出的最优的K个结果
        # 开始建立SIFT生成器
        descriptor = cv2.xfeatures2d.SIFT_create()
        # 其次检测特征点并计算描述子
        kps, features = descriptor.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, features # 这里返回特征点以及特征图

    def matchKeypoints(self, kpsA, kpsB, featureA, featureB, ratio, reprojThresh):# 函数用于匹配，找到匹配度较高的点
        matcher = cv2.BFMatcher()
        # 使用KNN检测来自AB图的SIFT特征匹配
        rawMatches = matcher.knnMatch(featureA, featureB, 2)
        # 这里是用于过滤的
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # 计算H矩阵，实际上所谓的转置矩阵 H，就是用来计算相对位置的。图像拼接的本质
            # 其实很简单，就是在一张图像合适的位置上拼接上另一张图片，而这个合适
            # 的位置就要去可以把重复的部分，也就是哪一块区域覆盖掉，而转置矩阵 H
            # 实际上就是用来定位的。
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return matches, H, status

    # 展示图像
    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # 返回可视化结果
        return vis
