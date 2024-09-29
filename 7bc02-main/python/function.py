# coding:utf8
import cv2
import cv2 as cv
import numpy as np
from PIL import Image
from numpy import fft
from pylab import *
from aip import AipOcr

save_path = "change.png"


# 图片写入文字
def Drawworld(img, text, p_x, p_y, font_type, font_size, bold, color):
    pos = (p_x, p_y)
    cv.putText(img, text, pos, font_type, font_size, color, bold)
    cv.imwrite(save_path, img)


# 空间转换
def color_space(img, c_type):
    if c_type == 1:
        out = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    elif c_type == 2:
        out = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif c_type == 3:
        out = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    elif c_type == 4:
        out = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    elif c_type == 5:
        out = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    cv.imwrite(save_path, out)
    return out


# 图像旋转

def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    Roa = cv.getRotationMatrix2D(center, angle, 1.0)
    out = cv.warpAffine(image, Roa, (w, h))
    cv.imwrite(save_path, out)
    return out


# 图像缩放
def changescale(image, size):
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    b2 = cv.resize(b, (0, 0), fx=size, fy=size, interpolation=cv.INTER_NEAREST)
    g2 = cv.resize(g, (0, 0), fx=size, fy=size, interpolation=cv.INTER_NEAREST)
    r2 = cv.resize(r, (0, 0), fx=size, fy=size, interpolation=cv.INTER_NEAREST)

    out = cv.merge([b2, g2, r2])

    cv.imwrite(save_path, out)
    return out


# 图像翻转
# def myflip(image , direction):
#     out = cv.flip(image, direction)
#     cv.imwrite(save_path, out)
#     return out

# 图像投影矫正
def correct(image):
    pts1 = np.float32([[158, 25], [267, 136], [58, 66], [144, 212]])
    # 变换后分别在左上、右上、左下、右下四个点
    pts2 = np.float32([[0, 0], [320, 0], [0, 200], [320, 200]])

    # 生成透视变换矩阵
    M = cv.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv.warpPerspective(image, M, (320, 200))

    out = dst
    cv.imwrite(save_path, out)
    return out


def horizontal_flip(I):
    I = cv2.flip(I, 1)
    return I


def vertical_flip(I):
    I = cv2.flip(I, 0)
    return I


def rotate_clockwise(I):
    dst = I
    for _ in range(3):
        dst = cv2.flip(dst, 1)  # 原型：cv2.flip(src, flipCode[, dst]) → dst  flipCode表示对称轴 0：x轴  1：y轴.  -1：both
        dst = cv2.transpose(dst)
    return dst


def rotate_anticlockwise(I):
    dst = cv2.flip(I, 1)  # 原型：cv2.flip(src, flipCode[, dst]) → dst  flipCode表示对称轴 0：x轴  1：y轴.  -1：both
    dst = cv2.transpose(dst)
    return dst


#二值化
def erzhihua(I):
    import cv2
    import numpy as np
    from PIL import Image
    from PIL import ImageEnhance
    from scipy.ndimage import gaussian_filter
    def getImageVar(img):
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
        return imageVar

    def gauss_division(image):
        src1 = image.astype(np.float32)
        gauss = gaussian_filter(image, sigma=101)
        gauss1 = gauss.astype(np.float32)
        dst1 = (src1 / gauss1)
        return dst1

    def image_enhancement(image):
        '''图像增强，增强对比度和亮度
        image PIL'''
        # 对比度增强
        im = Image.fromarray(np.uint8(image))
        enh_con = ImageEnhance.Contrast(im)
        # contrast = 5
        image_contrasted = enh_con.enhance(10)
        # 亮度增强
        enh_bri = ImageEnhance.Brightness(image_contrasted)
        image_contrasted1 = cv2.cvtColor(np.asarray(image_contrasted), cv2.COLOR_RGB2BGR)  # PIL转cv2
        clear = getImageVar(image_contrasted1)
        # print(clear)
        brightness = max(round(clear / 2000, 1), 1)
        # print(brightness)
        image_brightened = enh_bri.enhance(brightness)
        return image_brightened

    def ezh(I):
        hest = np.zeros([256], dtype=np.float)
        for row in range(h):
            for col in range(w):
                pv = (int)(I[row, col] + 0.5)
                hest[pv] += 1
        hest = hest / (h * w)
        T0 = 0
        for i in range(256):
            T0 = T0 + hest[i] * i
        T1den = 0
        T1 = 0
        T1num = 0
        for i in range((int)(T0 + 0.5)):
            T1den = T1den + hest[i]
            T1num = T1num + hest[i] * i
        T1 = T1num / T1den
        T1num = 0
        T1den = 0
        for i in range((int)(T0 + 0.5) + 1, 256):
            T1den = T1den + hest[i]
            T1num = T1num + hest[i] * i
        T1 = T1 + T1num / T1den
        T1 = T1 / 2
        while (abs(T1 - T0) > 0.0001):
            T0 = T1
            T1den = 0
            T1 = 0
            T1num = 0
            for i in range((int)(T0 + 0.5)):
                T1den = T1den + hest[i]
                T1num = T1num + hest[i] * i
            T1 = T1num / T1den
            T1num = 0
            T1den = 0
            for i in range((int)(T0 + 0.5) + 1, 256):
                T1den = T1den + hest[i]
                T1num = T1num + hest[i] * i
            T1 = T1 + T1num / T1den
            T1 = T1 / 2
        r, I = cv2.threshold(I, (int)(T1 + 0.5), 255, type=cv2.THRESH_BINARY)
        #cv2.imshow('out', I)
        #cv2.waitKey(0)
        return I

    # I = cv2.imread("testimg.jpg")
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    h, w = np.shape(I)
    #I=ezh(I)

    I = gauss_division(I)
    I = image_enhancement(I)
    I = np.array(I)
    for row in range(h):
        for col in range(w):
            if I[row, col] > 1:
                I[row, col] = 1
    I = I * 255
    '''
    cv2.imshow('out',I)
    cv2.waitKey(0)
    '''
    return I


#校正
def jiaozheng(I):
    try:
        # 输入原图，return矫正后的新图像
        # I=cv2.imread("binimg2.jpg",cv2.IMREAD_GRAYSCALE)
        I = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)[1]
        Icpy = I
        I = cv2.erode(I, np.ones((3, 3), np.uint8), iterations=1)
        N, M = I.shape[0], I.shape[1]
        p = np.zeros(N * M)

        for i in range(N * M):
            p[i] = i

        def find(x):
            x = int(x)
            if (x != p[x]):
                p[x] = find(p[x])
            return int(p[x])

        idx2area = {}
        area2idx = {}

        def merge():
            idx = 0
            for j in range(M):
                for i in range(N):
                    if I[i, j] == 0:
                        pos = i * M + j
                        if p[pos] not in area2idx:
                            temp = []
                            temp.append(pos)
                            idx2area[idx] = temp
                            area2idx[int(p[pos])] = idx
                            idx = idx + 1
                        else:
                            cur = area2idx[int(p[pos])]
                            temp = idx2area[cur]
                            temp.append(pos)
                            idx2area[cur] = temp
            return idx

        for j in range(1, M):
            for i in range(1, N):
                if I[i, j] == 0:
                    pos = i * M + j
                    if I[i - 1, j] == 0:
                        posl = (int)(i - 1) * M + j
                        ppos = find(pos)
                        pposl = find(posl)
                        if ppos != pposl:
                            p[ppos] = pposl
                    if I[i, j - 1] == 0:
                        posu = i * M + j - 1
                        ppos = find(pos)
                        pposu = find(int(posu))
                        if ppos != pposu:
                            p[int(ppos)] = int(pposu)
                    if I[i - 1, j - 1] == 0:
                        posu = (i - 1) * M + j - 1
                        ppos = find(pos)
                        pposu = find(posu)
                        if ppos != pposu:
                            p[ppos] = pposu
        for i in range(N * M):
            p[i] = find(i)
        num_of_areas = merge()

        # 第一次合并
        class rect:
            up = 0
            down = 0
            left = 0
            right = 0
            cx = 0
            cy = 0

            def __init__(self, left, right, up, down):
                self.up = up
                self.down = down
                self.left = left
                self.right = right
                self.cx = (left + right) // 2
                self.cy = (up + down) // 2

            def __lt__(self, x):
                if self.cx != x.cx:
                    return self.cx < x.cx
                return self.cy < x.cy

        R = []
        for area in idx2area.values():
            i = area[0]
            x, y = i % M, i // M
            P = rect(x, x, y, y)
            #    print(P.left,P.right,P.up,P.down,P.cx,P.cy)
            for i in area:
                x, y = i % M, i // M
                if x < P.left:
                    P.left = x
                if x > P.right:
                    P.right = x
                if y < P.up:
                    P.up = y
                if y > P.down:
                    P.down = y

            R.append(P)
        # for area in R:
        #    I=cv2.rectangle(I,(area.left,area.up),(area.right,area.down),(0,255,255),1)

        '''
        ttt=
        for i in idx2area[ttt]:
            x,y=i%M,i//M
            print(x,y)
            I=cv2.circle(I,(x,y),3,(0,0,255),-1)

        print(R[ttt].left,R[ttt].up,R[ttt].right,R[ttt].down)
        '''
        # 第二次合并
        Hsum = 0
        Hidx = 0
        for area in R:
            if area.down - area.up > 5:
                Hsum += area.down - area.up
                Hidx = Hidx + 1
        Hs = Hsum // Hidx
        Ws = Hs
        p_merged = [0] * (len(idx2area))
        for i in range(0, len(p_merged)):
            p_merged[i] = i;

        def find_merged(x):
            x = int(x)
            if (x != p_merged[x]):
                p_merged[x] = find(p_merged[x])
            return int(p_merged[x])

        for i in range(0, len(R)):
            # I=cv2.rectangle(I,(area.left,area.up),(area.right,area.down),(0,255,255),1)
            Win = rect(R[i].left, R[i].left + Ws, R[i].up, R[i].up + Hs)
            Wi, Hi = R[i].right - R[i].left, R[i].down - R[i].up
            for j in range(0, len(R)):
                Wj, Hj = R[j].right - R[j].left, R[j].down - R[j].up
                if i != j:
                    H_com = max(abs(R[i].down - R[j].up), abs(R[i].up - R[j].down))
                    W_com = max(abs(R[i].right - R[j].left), abs(R[j].right - R[i].left))
                    if W_com >= 1 and H_com >= 1:
                        if (Wi + Wj) / W_com >= 1 and (Hi + Hj) / H_com >= 1:
                            pi = find_merged(i)
                            pj = find_merged(j)
                            if (pi != pj):
                                p_merged[pi] = pj
                    if W_com >= 1:
                        if (Wi + Wj) / W_com >= 1 and 0.29 <= (Hi + Hj) / H_com and (
                                Hi + Hj) / H_com < 1 and W_com <= Ws and H_com <= Hs:
                            pi = find_merged(i)
                            pj = find_merged(j)
                            if (pi != pj):
                                p_merged[pi] = pj
        idx2area_merged = {}
        area2idx_merged = {}
        idx_merged = 0
        for i in range(len(idx2area)):
            if p_merged[i] not in area2idx_merged:
                temp = []
                temp.append(i)
                idx2area_merged[idx_merged] = temp
                area2idx_merged[int(p_merged[i])] = idx_merged
                idx_merged = idx_merged + 1
            else:
                cur = area2idx_merged[int(p_merged[i])]
                temp = idx2area_merged[cur]
                temp.append(i)
                idx2area_merged[cur] = temp
        R_merged = []
        for area in idx2area_merged.values():
            i = R[area[0]]
            x, y = i.left, i.up
            P = rect(x, x, y, y)
            for i in area:
                x, y, z, w = R[i].left, R[i].up, R[i].right, R[i].down
                if x < P.left:
                    P.left = x
                if z > P.right:
                    P.right = z
                if y < P.up:
                    P.up = y
                if w > P.down:
                    P.down = w
            P.cx = (P.left + P.right) // 2
            P.cy = (P.up + P.down) // 2
            R_merged.append(P)
        # for area in R_merged:
        #    Icpy=cv2.rectangle(Icpy,(area.left,area.up),(area.right,area.down),(0,255,255),1)
        # ttt=153
        # Icpy=cv2.rectangle(Icpy,(R_merged[ttt].left,R_merged[ttt].up),(R_merged[ttt].right,R_merged[ttt].down),(0,255,255),1)
        hangliantongyu = []
        R_merged.sort()
        line_next = [-1] * (len(idx2area_merged))
        line_pre = [-1] * (len(idx2area_merged))
        for i in range(len(R_merged)):
            nextpos = -1
            for j in range(i + 1, len(R_merged)):
                if not ((R_merged[i].up > R_merged[j].down) or R_merged[i].down < R_merged[j].up):
                    # print(R_merged[i].up,R_merged[i].down,R_merged[j].up,R_merged[j].down)
                    if nextpos == -1:
                        nextpos = j
                    elif (R_merged[j].cx - R_merged[i].cx < R_merged[nextpos].cx - R_merged[i].cx):
                        nextpos = j
            if nextpos != -1:
                line_next[i] = nextpos
                line_pre[nextpos] = i
        idx2area_line = {}  # 根据字找行
        idx_line = 0
        for i in range(len(R_merged)):
            if line_pre[i] == -1:
                temp = []
                temp.append(i)
                idx2area_line[i] = idx_line
                idx_line = idx_line + 1
            else:
                idx2area_line[i] = idx2area_line[line_pre[i]]
        for i in range(len(R_merged)):
            if line_pre[i] == -1:
                prepos = -1
                for j in range(0, i - 1):
                    if not ((R_merged[j].up > R_merged[i].down) or R_merged[j].down < R_merged[i].up):
                        # print(R_merged[i].up,R_merged[i].down,R_merged[j].up,R_merged[j].down)
                        if prepos == -1:
                            prepos = j
                        elif (R_merged[i].cx - R_merged[j].cx < R_merged[i].cx - R_merged[prepos].cx):
                            prepos = j
                if prepos != -1:
                    line_pre[i] = prepos
        R_line = []
        for i in range(idx_line):
            R_line.append([])
        for i in range(len(R_merged)):
            temp = R_line[idx2area_line[i]]
            temp.append(i)
            R_line[idx2area_line[i]] = temp
        newimg = np.ones((N, M), dtype=np.uint8) * 255
        for lin in range(idx_line):
            c_m = R_merged[R_line[lin][0]].cy
            for k in range(len(R_line[lin])):
                if R_merged[R_line[lin][k]].down - R_merged[R_line[lin][k]].up < Hs / 2:
                    d = R_merged[R_line[lin][k]].down - R_merged[R_line[lin][0]].down
                else:
                    c_n = R_merged[R_line[lin][k]].cy
                    d = c_n - c_m
                for i in range(R_merged[R_line[lin][k]].up, R_merged[R_line[lin][k]].down + 1):
                    for j in range(R_merged[R_line[lin][k]].left, R_merged[R_line[lin][k]].right + 1):
                        # print(i,j,d)
                        if i - d >= N:
                            continue
                        else:
                            newimg[i - d, j] = Icpy[i, j]

        return newimg
    except Exception as e:
        return I


# def jiaozheng(I):
#     # 输入原图，return矫正后的新图像
#     # I=cv2.imread("binimg2.jpg",cv2.IMREAD_GRAYSCALE)
#     I = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)[1]
#     Icpy = I
#     I = cv2.erode(I, np.ones((3, 3), np.uint8), iterations=1)
#     N, M = I.shape[0],I.shape[1]
#     p = np.zeros(N * M)
#
#     for i in range(N * M):
#         p[i] = i
#
#     def find(x):
#         x = int(x)
#         if (x != p[x]):
#             p[x] = find(p[x])
#         return int(p[x])
#
#     idx2area = {}
#     area2idx = {}
#
#     def merge():
#         idx = 0
#         for j in range(M):
#             for i in range(N):
#                 if I[i, j] == 0:
#                     pos = i * M + j
#                     if p[pos] not in area2idx:
#                         temp = []
#                         temp.append(pos)
#                         idx2area[idx] = temp
#                         area2idx[int(p[pos])] = idx
#                         idx = idx + 1
#                     else:
#                         cur = area2idx[int(p[pos])]
#                         temp = idx2area[cur]
#                         temp.append(pos)
#                         idx2area[cur] = temp
#         return idx
#
#     for j in range(1, M):
#         for i in range(1, N):
#             if I[i, j] == 0:
#                 pos = i * M + j
#                 if I[i - 1, j] == 0:
#                     posl = (int)(i - 1) * M + j
#                     ppos = find(pos)
#                     pposl = find(posl)
#                     if ppos != pposl:
#                         p[ppos] = pposl
#                 if I[i, j - 1] == 0:
#                     posu = i * M + j - 1
#                     ppos = find(pos)
#                     pposu = find(int(posu))
#                     if ppos != pposu:
#                         p[int(ppos)] = int(pposu)
#                 if I[i - 1, j - 1] == 0:
#                     posu = (i - 1) * M + j - 1
#                     ppos = find(pos)
#                     pposu = find(posu)
#                     if ppos != pposu:
#                         p[ppos] = pposu
#     for i in range(N * M):
#         p[i] = find(i)
#     num_of_areas = merge()
#
#     # 第一次合并
#     class rect:
#         up = 0
#         down = 0
#         left = 0
#         right = 0
#         cx = 0
#         cy = 0
#
#         def __init__(self, left, right, up, down):
#             self.up = up
#             self.down = down
#             self.left = left
#             self.right = right
#             self.cx = (left + right) // 2
#             self.cy = (up + down) // 2
#
#         def __lt__(self, x):
#             if self.cx != x.cx:
#                 return self.cx < x.cx
#             return self.cy < x.cy
#
#     R = []
#     for area in idx2area.values():
#         i = area[0]
#         x, y = i % M, i // M
#         P = rect(x, x, y, y)
#         #    print(P.left,P.right,P.up,P.down,P.cx,P.cy)
#         for i in area:
#             x, y = i % M, i // M
#             if x < P.left:
#                 P.left = x
#             if x > P.right:
#                 P.right = x
#             if y < P.up:
#                 P.up = y
#             if y > P.down:
#                 P.down = y
#
#         R.append(P)
#     # for area in R:
#     #    I=cv2.rectangle(I,(area.left,area.up),(area.right,area.down),(0,255,255),1)
#
#     '''
#     ttt=
#     for i in idx2area[ttt]:
#         x,y=i%M,i//M
#         print(x,y)
#         I=cv2.circle(I,(x,y),3,(0,0,255),-1)
#
#     print(R[ttt].left,R[ttt].up,R[ttt].right,R[ttt].down)
#     '''
#     # 第二次合并
#     Hsum = 0
#     Hidx = 0
#     for area in R:
#         if area.down - area.up > 5:
#             Hsum += area.down - area.up
#             Hidx = Hidx + 1
#     Hs = Hsum // Hidx
#     Ws = Hs
#     p_merged = [0] * (len(idx2area))
#     for i in range(0, len(p_merged)):
#         p_merged[i] = i;
#
#     def find_merged(x):
#         x = int(x)
#         if (x != p_merged[x]):
#             p_merged[x] = find(p_merged[x])
#         return int(p_merged[x])
#
#     for i in range(0, len(R)):
#         # I=cv2.rectangle(I,(area.left,area.up),(area.right,area.down),(0,255,255),1)
#         Win = rect(R[i].left, R[i].left + Ws, R[i].up, R[i].up + Hs)
#         Wi, Hi = R[i].right - R[i].left, R[i].down - R[i].up
#         for j in range(0, len(R)):
#             Wj, Hj = R[j].right - R[j].left, R[j].down - R[j].up
#             if i != j:
#                 H_com = max(abs(R[i].down - R[j].up), abs(R[i].up - R[j].down))
#                 W_com = max(abs(R[i].right - R[j].left), abs(R[j].right - R[i].left))
#                 if W_com >= 1 and H_com >= 1:
#                     if (Wi + Wj) / W_com >= 1 and (Hi + Hj) / H_com >= 1:
#                         pi = find_merged(i)
#                         pj = find_merged(j)
#                         if (pi != pj):
#                             p_merged[pi] = pj
#                 if W_com >= 1:
#                     if (Wi + Wj) / W_com >= 1 and 0.29 <= (Hi + Hj) / H_com and (
#                             Hi + Hj) / H_com < 1 and W_com <= Ws and H_com <= Hs:
#                         pi = find_merged(i)
#                         pj = find_merged(j)
#                         if (pi != pj):
#                             p_merged[pi] = pj
#     idx2area_merged = {}
#     area2idx_merged = {}
#     idx_merged = 0
#     for i in range(len(idx2area)):
#         if p_merged[i] not in area2idx_merged:
#             temp = []
#             temp.append(i)
#             idx2area_merged[idx_merged] = temp
#             area2idx_merged[int(p_merged[i])] = idx_merged
#             idx_merged = idx_merged + 1
#         else:
#             cur = area2idx_merged[int(p_merged[i])]
#             temp = idx2area_merged[cur]
#             temp.append(i)
#             idx2area_merged[cur] = temp
#     R_merged = []
#     for area in idx2area_merged.values():
#         i = R[area[0]]
#         x, y = i.left, i.up
#         P = rect(x, x, y, y)
#         for i in area:
#             x, y, z, w = R[i].left, R[i].up, R[i].right, R[i].down
#             if x < P.left:
#                 P.left = x
#             if z > P.right:
#                 P.right = z
#             if y < P.up:
#                 P.up = y
#             if w > P.down:
#                 P.down = w
#         P.cx = (P.left + P.right) // 2
#         P.cy = (P.up + P.down) // 2
#         R_merged.append(P)
#     # for area in R_merged:
#     #    Icpy=cv2.rectangle(Icpy,(area.left,area.up),(area.right,area.down),(0,255,255),1)
#     # ttt=153
#     # Icpy=cv2.rectangle(Icpy,(R_merged[ttt].left,R_merged[ttt].up),(R_merged[ttt].right,R_merged[ttt].down),(0,255,255),1)
#     hangliantongyu = []
#     R_merged.sort()
#     line_next = [-1] * (len(idx2area_merged))
#     line_pre = [-1] * (len(idx2area_merged))
#     for i in range(len(R_merged)):
#         nextpos = -1
#         for j in range(i + 1, len(R_merged)):
#             if not ((R_merged[i].up > R_merged[j].down) or R_merged[i].down < R_merged[j].up):
#                 # print(R_merged[i].up,R_merged[i].down,R_merged[j].up,R_merged[j].down)
#                 if nextpos == -1:
#                     nextpos = j
#                 elif (R_merged[j].cx - R_merged[i].cx < R_merged[nextpos].cx - R_merged[i].cx):
#                     nextpos = j
#         if nextpos != -1:
#             line_next[i] = nextpos
#             line_pre[nextpos] = i
#     idx2area_line = {}  # 根据字找行
#     idx_line = 0
#     for i in range(len(R_merged)):
#         if line_pre[i] == -1:
#             temp = []
#             temp.append(i)
#             idx2area_line[i] = idx_line
#             idx_line = idx_line + 1
#         else:
#             idx2area_line[i] = idx2area_line[line_pre[i]]
#     for i in range(len(R_merged)):
#         if line_pre[i] == -1:
#             prepos = -1
#             for j in range(0, i - 1):
#                 if not ((R_merged[j].up > R_merged[i].down) or R_merged[j].down < R_merged[i].up):
#                     # print(R_merged[i].up,R_merged[i].down,R_merged[j].up,R_merged[j].down)
#                     if prepos == -1:
#                         prepos = j
#                     elif (R_merged[i].cx - R_merged[j].cx < R_merged[i].cx - R_merged[prepos].cx):
#                         prepos = j
#             if prepos != -1:
#                 line_pre[i] = prepos
#     R_line = []
#     for i in range(idx_line):
#         R_line.append([])
#     for i in range(len(R_merged)):
#         temp = R_line[idx2area_line[i]]
#         temp.append(i)
#         R_line[idx2area_line[i]] = temp
#     newimg = np.ones((N, M), dtype=np.uint8) * 255
#     for lin in range(idx_line):
#         c_m = R_merged[R_line[lin][0]].cy
#         for k in range(len(R_line[lin])):
#             if R_merged[R_line[lin][k]].down - R_merged[R_line[lin][k]].up < Hs / 2:
#                 d = R_merged[R_line[lin][k]].down - R_merged[R_line[lin][0]].down
#             else:
#                 c_n = R_merged[R_line[lin][k]].cy
#                 d = c_n - c_m
#             for i in range(R_merged[R_line[lin][k]].up, R_merged[R_line[lin][k]].down + 1):
#                 for j in range(R_merged[R_line[lin][k]].left, R_merged[R_line[lin][k]].right + 1):
#                     #print(i,j,d)
#                     if i-d>=N:
#                         continue
#                     else:
#                         newimg[i - d, j] = Icpy[i, j]
#
#     return newimg

#纹理平滑1
def wenlipinghua(I):
    def psf2otf_Dy(outSize=None):
        psf = np.zeros((outSize), dtype='float32')
        psf[0, 0] = - 1
        psf[-1, 0] = 1
        otf = np.fft.fft2(psf)
        return otf

    def psf2otf_Dx(outSize=None):
        psf = np.zeros((outSize), dtype='float32')
        psf[0, 0] = - 1
        psf[0, -1] = 1
        otf = np.fft.fft2(psf)
        return otf

    def dowenlipinghua(img, lambda_=1, iter=4, p=0.8):
        eps = 0.0001
        #img=cv2.imread('test2img.jpg')
        #cv2.imshow('input',img)
        Img = np.array(img, dtype='float64')
        Img = Img / 255
        gamma = 0.5 * p - 1
        c = p * eps ** gamma
        N, M, D = Img.shape
        sizeI2D = np.array([N, M])
        otfFx = psf2otf_Dx(sizeI2D)
        otfFy = psf2otf_Dy(sizeI2D)
        Denormin = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
        Denormin = np.tile(Denormin, (D, 1, 1))
        Denormin = Denormin.transpose((1, 2, 0))
        Denormin = 1 + 0.5 * c * lambda_ * Denormin
        U = Img
        Normin1 = np.zeros((N, M, D), dtype=complex)
        for _dim in range(D):
            Normin1[:, :, _dim] = np.fft.fft2(U[:, :, _dim])
        for k in range(1, iter + 1):
            # Intermediate variables \mu update, in x-axis and y-axis direction
            u01 = U[:, 0, :] - U[:, -1, :]
            u01 = np.reshape(u01, (u01.shape[0], 1, u01.shape[1]))
            u_h = np.concatenate((np.diff(U, 1, 1), u01), axis=1)
            u10 = U[0, :, :] - U[-1, :, :]
            u10 = np.reshape(u10, (1, u10.shape[0], u10.shape[1]))
            u_v = np.concatenate((np.diff(U, 1, 0), u10), axis=0)
            mu_h = np.multiply(c, u_h) - np.multiply(np.multiply(p, u_h), ((np.multiply(u_h, u_h) + eps) ** gamma))
            mu_v = np.multiply(c, u_v) - np.multiply(np.multiply(p, u_v), ((np.multiply(u_v, u_v) + eps) ** gamma))
            # Update the smoothed image U
            mu_h01 = mu_h[:, -1, :] - mu_h[:, 0, :]
            mu_h01 = np.reshape(mu_h01, (mu_h01.shape[0], 1, mu_h01.shape[1]))
            Normin2_h = np.concatenate((mu_h01, - np.diff(mu_h, 1, 1)), axis=1)
            mu_v01 = mu_v[-1, :, :] - mu_v[0, :, :]
            mu_v01 = np.reshape(mu_v01, (1, mu_v01.shape[0], mu_v01.shape[1]))
            Normin2_v = np.concatenate((mu_v01, - np.diff(mu_v, 1, 0)), axis=0)
            FU = np.zeros((N, M, D), dtype=complex)
            for _dim in range(D):
                FU[:, :, _dim] = (Normin1[:, :, _dim] + 0.5 * lambda_ * (
                    np.fft.fft2(Normin2_h[:, :, _dim] + Normin2_v[:, :, _dim]))) / Denormin[:, :, _dim]
            U = np.zeros((N, M, D))
            for _dim in range(D):
                U[:, :, _dim] = np.real(np.fft.ifft2(FU[:, :, _dim]))
            Normin1 = FU
        # Smoothed = ILS_LNorm_GPU(Img, lambda, p, eps, iter);
        Smoothed = U
        return Smoothed

    S = dowenlipinghua(I)
    S = S * 255
    return S.astype(np.uint8)


#纹理平滑2
def L0Smoothing(I):
    def psf2otf_Dy(outSize=None):
        psf = np.zeros((outSize), dtype='float32')
        psf[0, 0] = - 1
        psf[-1, 0] = 1
        otf = np.fft.fft2(psf)
        return otf

    def psf2otf_Dx(outSize=None):
        psf = np.zeros((outSize), dtype='float32')
        psf[0, 0] = - 1
        psf[0, -1] = 1
        otf = np.fft.fft2(psf)
        return otf

    def doL0Smoothing(Im=None, lambda_=None, kappa=None):
        # L0Smooth - Image Smoothing via L0 Gradient Minimization
        #   S = L0Smooth(Im, lambda, kappa) performs L0 graidient smoothing of input
        #   image Im, with smoothness weight lambda and rate kappa.

        #   Paras:
        #   @Im    : Input UINT8 image, both grayscale and color images are acceptable.
        #   @lambda: Smoothing parameter controlling the degree of smooth. (See [1])
        #            Typically it is within the range [1e-3, 1e-1], 2e-2 by default.
        #   @kappa : Parameter that controls the rate. (See [1])
        #            Small kappa results in more iteratioins and with sharper edges.
        #            We select kappa in (1, 2].
        #            kappa = 2 is suggested for natural images.

        #   Example
        #   ==========
        #   Im  = imread('pflower.jpg');
        #   S  = L0Smooth(Im); # Default Parameters (lambda = 2e-2, kappa = 2)
        #   figure, imshow(Im), figure, imshow(S);
        S = Im
        if (kappa == None):
            kappa = 2.0

        if (lambda_ == None):
            lambda_ = 0.02

        betamax = 100000.0
        fx = np.array([1, - 1])
        fy = np.array([[1], [- 1]])
        N, M, D = Im.shape
        sizeI3D = np.array([N, M, D])
        sizeI2D = np.array([N, M])
        otfFx = psf2otf_Dx(sizeI2D)
        otfFy = psf2otf_Dy(sizeI2D)
        Normin1 = np.zeros((N, M, D), dtype=complex)
        for _dim in range(D):
            Normin1[:, :, _dim] = np.fft.fft2(Im[:, :, _dim])
        Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
        if D > 1:
            Denormin2 = np.tile(Denormin2, (D, 1, 1))
            Denormin2 = Denormin2.transpose((1, 2, 0))

        beta = 2 * lambda_
        while beta < betamax:

            Denormin = 1 + beta * Denormin2
            # h-v subproblem
            u01 = Im[:, 0, :] - Im[:, -1, :]
            u01 = np.reshape(u01, (u01.shape[0], 1, u01.shape[1]))
            h = np.concatenate((np.diff(Im, 1, 1), u01), axis=1)
            u10 = Im[0, :, :] - Im[-1, :, :]
            u10 = np.reshape(u10, (1, u10.shape[0], u10.shape[1]))
            v = np.concatenate((np.diff(Im, 1, 0), u10), axis=0)
            if D == 1:
                t = (h ** 2 + v ** 2) < lambda_ / beta
            else:
                t = np.sum((h ** 2 + v ** 2), 3 - 1) < lambda_ / beta
                t = np.tile(t, (D, 1, 1))
                t = t.transpose((1, 2, 0))
                #t = np.matlib.repmat(t, np.array([1, 1, D]))
            h[t] = 0
            v[t] = 0
            # S subproblem
            mu_h01 = h[:, -1, :] - h[:, 0, :]
            mu_h01 = np.reshape(mu_h01, (mu_h01.shape[0], 1, mu_h01.shape[1]))
            Normin2_h = np.concatenate((mu_h01, - np.diff(h, 1, 1)), axis=1)
            mu_v01 = v[-1, :, :] - v[0, :, :]
            mu_v01 = np.reshape(mu_v01, (1, mu_v01.shape[0], mu_v01.shape[1]))
            Normin2_v = np.concatenate((mu_v01, - np.diff(v, 1, 0)), axis=0)
            Normin2 = Normin2_h + Normin2_v
            FS = np.zeros((N, M, D), dtype=complex)
            for _dim in range(D):
                FS[:, :, _dim] = (Normin1[:, :, _dim] + beta * (np.fft.fft2(Normin2[:, :, _dim]))) / Denormin[:, :,
                                                                                                     _dim]
            #S=np.zeros((N,M,D))
            for _dim in range(D):
                S[:, :, _dim] = np.real(np.fft.ifft2(FS[:, :, _dim]))
            #        FS = (Normin1 + beta * fft2(Normin2)) / Denormin
            #        S = real(ifft2(FS))
            beta = beta * kappa
        return S

    I = np.array(I, dtype='float64')
    I = I / 255
    S = doL0Smoothing(I, 0.01)
    S = S * 255
    return S.astype(np.uint8)


#纹理增强
def wenlizengqiang(I):
    def psf2otf_Dy(outSize=None):
        psf = np.zeros((outSize), dtype='float32')
        psf[0, 0] = - 1
        psf[-1, 0] = 1
        otf = np.fft.fft2(psf)
        return otf

    def psf2otf_Dx(outSize=None):
        psf = np.zeros((outSize), dtype='float32')
        psf[0, 0] = - 1
        psf[0, -1] = 1
        otf = np.fft.fft2(psf)
        return otf

    def dowenlizengqiang(img, lambda_=1, iter=4, p=0.8):
        eps = 0.0001
        #img=cv2.imread('test2img.jpg')
        #cv2.imshow('input',img)
        Img = np.array(img, dtype='float64')
        Img = Img / 255
        gamma = 0.5 * p - 1
        c = p * eps ** gamma
        N, M, D = Img.shape
        sizeI2D = np.array([N, M])
        otfFx = psf2otf_Dx(sizeI2D)
        otfFy = psf2otf_Dy(sizeI2D)
        Denormin = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
        Denormin = np.tile(Denormin, (D, 1, 1))
        Denormin = Denormin.transpose((1, 2, 0))
        Denormin = 1 + 0.5 * c * lambda_ * Denormin
        U = Img

        Normin1 = np.zeros((N, M, D), dtype=complex)
        for _dim in range(D):
            Normin1[:, :, _dim] = np.fft.fft2(U[:, :, _dim])
        for k in range(1, iter + 1):
            # Intermediate variables \mu update, in x-axis and y-axis direction
            u01 = U[:, 0, :] - U[:, -1, :]
            u01 = np.reshape(u01, (u01.shape[0], 1, u01.shape[1]))
            u_h = np.concatenate((np.diff(U, 1, 1), u01), axis=1)
            u10 = U[0, :, :] - U[-1, :, :]
            u10 = np.reshape(u10, (1, u10.shape[0], u10.shape[1]))
            u_v = np.concatenate((np.diff(U, 1, 0), u10), axis=0)
            mu_h = np.multiply(c, u_h) - np.multiply(np.multiply(p, u_h), ((np.multiply(u_h, u_h) + eps) ** gamma))
            mu_v = np.multiply(c, u_v) - np.multiply(np.multiply(p, u_v), ((np.multiply(u_v, u_v) + eps) ** gamma))
            # Update the smoothed image U
            mu_h01 = mu_h[:, -1, :] - mu_h[:, 0, :]
            mu_h01 = np.reshape(mu_h01, (mu_h01.shape[0], 1, mu_h01.shape[1]))
            Normin2_h = np.concatenate((mu_h01, - np.diff(mu_h, 1, 1)), axis=1)
            mu_v01 = mu_v[-1, :, :] - mu_v[0, :, :]
            mu_v01 = np.reshape(mu_v01, (1, mu_v01.shape[0], mu_v01.shape[1]))
            Normin2_v = np.concatenate((mu_v01, - np.diff(mu_v, 1, 0)), axis=0)
            FU = np.zeros((N, M, D), dtype=complex)
            for _dim in range(D):
                FU[:, :, _dim] = (Normin1[:, :, _dim] + 0.5 * lambda_ * (
                    np.fft.fft2(Normin2_h[:, :, _dim] + Normin2_v[:, :, _dim]))) / Denormin[:, :, _dim]
            U = np.zeros((N, M, D))
            for _dim in range(D):
                U[:, :, _dim] = np.real(np.fft.ifft2(FU[:, :, _dim]))
            Normin1 = FU
        # Smoothed = ILS_LNorm_GPU(Img, lambda, p, eps, iter);
        Smoothed = U
        Diff = Img - Smoothed
        # ImgE = Img + 3 * Diff
        ImgE = Img
        ImgE = np.uint8(ImgE * 255)
        Diff = np.uint8(Diff * 255)
        ImgE = cv2.add(ImgE, Diff)
        #ImgE=cv2.add(ImgE,Diff)
        #ImgE=cv2.add(ImgE,Diff)
        #        cv2.imshow('out',ImgE)
        return ImgE

    S = dowenlizengqiang(I)
    S = S.astype(np.uint8)
    #    cv2.imshow('imge2',S)
    return S.astype(np.uint8)


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")
    pts = points.sum(axis=1)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# def wrap (I,points):
#     src = order_points(points)
#     (tl, tr, br, bl) = src
#     widthA = np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
#     widthB = np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
#     maxW = int(max(widthA, widthB))
#     heightA = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
#     heightB = np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
#     maxH = int(max(heightA, heightB))
#     dst = np.array([[0,0],
#                    [maxW-1,0],
#                    [maxW-1,maxH-1],
#                    [0,maxH-1]],dtype="float32")
#     M = cv2.getPerspectiveTransform(src, dst)
#     res = cv2.warpPerspective(I, M,(maxW,maxH))
#     return res


def sort_points(coordinate):
    coordinate = np.array(coordinate)
    center = coordinate[0]
    for _ in range(1, 4):
        center = center + coordinate[_]
    center = center / 4
    coordinate_temp = coordinate.copy()  # 复制一份坐标，避免原坐标被破坏
    left_coordinate = []  # 存储x轴小于中心坐标点的点
    delete_index = []

    # 将 x轴小于中心坐标点的点 存储进left_coordinate
    for _ in range(4):
        if (coordinate[_][0] < center[0]):
            left_coordinate.append(coordinate[_])
            delete_index.append(_)
    # 将存储进 left_coordinate 的元素从coordinate_temp中删除
    coordinate_temp = np.delete(coordinate_temp, delete_index, axis=0)
    left_coordinate_temp = left_coordinate.copy()  # 避免程序过程因为left_coordinate的变动而导致最初的条件判断错误

    if (len(left_coordinate_temp) == 2):
        # 比较左边两个点的y轴，大的为左上
        if (left_coordinate[0][1] < left_coordinate[1][1]):
            left_bottom = left_coordinate[0]
            left_top = left_coordinate[1]
        elif (left_coordinate[0][1] > left_coordinate[1][1]):
            left_bottom = left_coordinate[1]
            left_top = left_coordinate[0]

        # 比较右边两个点的y轴，大的为右上
        if (coordinate_temp[0][1] < coordinate_temp[1][1]):
            right_bottom = coordinate_temp[0]
            right_top = coordinate_temp[1]
        elif (coordinate_temp[0][1] > coordinate_temp[1][1]):
            right_bottom = coordinate_temp[1]
            right_top = coordinate_temp[0]

    elif (len(left_coordinate_temp) == 1):
        left_bottom = left_coordinate[0]
        delete_index = []

        for _ in range(3):
            if (coordinate_temp[_][0] == center[0] and coordinate_temp[_][1] > center[1]):
                left_top = coordinate_temp[_]
                delete_index.append(_)
            if (coordinate_temp[_][0] == center[0] and coordinate_temp[_][1] < center[1]):
                right_bottom = coordinate_temp[_]
                delete_index.append(_)

        coordinate_temp = np.delete(coordinate_temp, delete_index, axis=0)
        right_top = coordinate_temp[0]
    res = np.array([left_bottom, right_bottom, left_top, right_top])
    # print(res)
    return res


def wrap(img, points):
    #输入原图和4*2的numpy数组，输出新图
    points = np.asarray(points, dtype='float32')
    points = sort_points(points)
    rows = img.shape[0]
    cols = img.shape[1]
    #print(rows)
    #print(cols)
    src = points
    dst = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (cols, rows))
    # print(out.shape[0])
    # print(out.shape[1])
    return out


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def ocr(img):
    api_key = 'ihWkKwAuyq7Ps95QwFXh7yqp'
    app_id = '23078413'
    secret_key = 'aSLAkCdphzqhV5Q8fBPIerPS0YsfBbYb'
    client = AipOcr(app_id, api_key, secret_key)
    file = open(r'output.txt', 'w', encoding='utf-8')
    cv2.imwrite('tempimg.jpg', img)
    i = get_file_content('tempimg.jpg')
    inf = client.basicGeneral(i)
    res = ""
    for response in inf['words_result']:
        for words in response['words']:
            res = res + words
        #print('\n')
    #print(inf)
    file.write(res)
    file.close()
    #print(res)
    return res
