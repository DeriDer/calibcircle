import cv2
import numpy as np
import os

def save_XML(retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F):
    # 创建 FileStorage 对象并打开 XML 文件
    calibration_data = cv2.FileStorage('stereo_calibration_data.xml', cv2.FILE_STORAGE_WRITE)

    # 写入相机参数和畸变系数到 XML 文件
    calibration_data.write("M1", cameraMatrix1)
    calibration_data.write("D1", distCoeffs1)
    calibration_data.write("M2", cameraMatrix2)
    calibration_data.write("D2", distCoeffs2)
    calibration_data.write("R", R)
    calibration_data.write("T", T)
    calibration_data.write("E", E)
    calibration_data.write("F", F)

    # 关闭 FileStorage 对象
    calibration_data.release()

def read_XML(file):
    # 加载相机标定结果
    calibration_data = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)

    # 读取相机参数和畸变系数
    camera_matrix_left = calibration_data.getNode("M1").mat()
    dist_coeffs_left = calibration_data.getNode("D1").mat()
    camera_matrix_right = calibration_data.getNode("M2").mat()
    dist_coeffs_right = calibration_data.getNode("D2").mat()
    rotation_matrix = calibration_data.getNode("R").mat()
    translation_vector = calibration_data.getNode("T").mat()
    E= calibration_data.getNode("E").mat()
    F= calibration_data.getNode("F").mat()

    # 关闭文件
    calibration_data.release()
    return camera_matrix_left,dist_coeffs_left,camera_matrix_right,dist_coeffs_right,rotation_matrix,translation_vector,E,F

def reprojection_error(world_points,rotation_matrix, translation_vector, camera_matrix_left, dist_coeffs_left, imagepoints_left):
    # 计算投影误差
    reprojection_errors = []
    for i in range(len(world_points)):
        # 投影到左相机图像平面上
        projected_point_left, _ = cv2.projectPoints(world_points[i], rotation_matrix, translation_vector,
                                                    camera_matrix_left, dist_coeffs_left)
        projected_point_left = projected_point_left.squeeze()

        # 获取实际观测到的左相机图像点
        observed_point_left = imagepoints_left[i]

        # 计算投影误差（欧氏距离）
        reprojection_error = np.linalg.norm(observed_point_left - projected_point_left)
        reprojection_errors.append(reprojection_error)

    # 计算平均冲投影误差
    mean_reprojection_error = np.mean(reprojection_errors)
    return mean_reprojection_error

def resize_image(image, width =684):
    # 获取原始图像的宽度和高度
    h, w = image.shape[:2]
    ratio = 5472 / width
    # 计算调整后的高度，以保持纵横比
    aspect_ratio = float(width) / w
    height = int(h * aspect_ratio)

    # 调整图像大小
    resized_image = cv2.resize(image, (width, height))

    return resized_image, ratio

def get_image_list(num_images = 13):
    lfiles =[]
    rfiles =[]
    for i in range(1, num_images + 1):
        filenamer = "gray_Right{}.bmp".format(i)
        rfiles.append(filenamer)
        
        filenamel = "gray_Left{}.bmp".format(i)
        lfiles.append(filenamel)
    
    return lfiles,rfiles

def calibrate_camera_stereo(image_folder_path, pattern_size, square_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备棋盘格的世界坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # 存储棋盘格角点的世界坐标和图像坐标
    objpoints = []  # 世界坐标系中的三维点
    limgpoints = []  # 图像平面中的二维点
    rimgpoints = []  # 图像平面中的二维点
    
    lfilelist =[]
    rfilelist =[]

    lfilelist, rfilelist = get_image_list()

    for lfile, rfile in zip(lfilelist, rfilelist):

        file_pathl = os.path.join(image_folder_path, lfile)
        file_pathr = os.path.join(image_folder_path, rfile)
        
        limg = cv2.imread(file_pathl)
        show_limg,lratio = resize_image(limg)
        lgrayori = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
        lgray,_ = resize_image(lgrayori)

        rimg = cv2.imread(file_pathr)
        show_rimg, rratio = resize_image(rimg)
        rgrayori = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
        rgray,_ = resize_image(rgrayori)
        #breakpoint()
        retl, cornersl = cv2.findChessboardCorners(lgray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        retr, cornersr = cv2.findChessboardCorners(rgray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # 如果找到了棋盘格角点
        if retl == True and retr == True:
            objpoints.append(objp)

            # 进行亚像素级角点检测
            lcorners2 = cv2.cornerSubPix(lgray, cornersl, (11, 11), (-1, -1), criteria)
            limgpoints.append(lcorners2)#*lratio)

            rcorners2 = cv2.cornerSubPix(rgray, cornersr, (11, 11), (-1, -1), criteria)
            rimgpoints.append(rcorners2)#*rratio)
            # 绘制并显示棋盘格角点
            show_rimg = cv2.drawChessboardCorners(show_rimg, pattern_size, rcorners2, retr)
            show_limg = cv2.drawChessboardCorners(show_limg, pattern_size, lcorners2, retl)
            result = cv2.hconcat([show_rimg, show_limg])
            cv2.imshow('img', result)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # 进行相机标定
    lret, lmtx, ldist, lrvecs, ltvecs = cv2.calibrateCamera(objpoints, limgpoints, lgray.shape[::-1], None, None)

    print("left相机标定结果：")
    print("相机矩阵：\n", lmtx)
    print("畸变系数：\n", ldist)

    lmean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], lrvecs[i], ltvecs[i], lmtx, ldist)
        error = cv2.norm(limgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        lmean_error += error
    print( "left total error: {}".format(lmean_error/len(objpoints)) )


    rret, rmtx, rdist, rrvecs, rtvecs = cv2.calibrateCamera(objpoints, rimgpoints, rgray.shape[::-1], None, None)

    print("right相机标定结果：")
    print("相机矩阵：\n", rmtx)
    print("畸变系数：\n", rdist)

    rmean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rrvecs[i], rtvecs[i], rmtx, rdist)
        error = cv2.norm(rimgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        rmean_error += error
    print( "right total error: {}".format(rmean_error/len(objpoints)) )

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    stereoflag = 0
    stereoflag |= cv2.CALIB_FIX_INTRINSIC
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, limgpoints, rimgpoints,
    lmtx, ldist, rmtx, rdist,
    lgray.shape[::-1], criteria= stereocalib_criteria, flags = stereoflag)   
    # 输出结果
    print("相机标定结果：")
    print("左侧相机矩阵：\n", cameraMatrix1)
    print("左侧畸变系数：\n", distCoeffs1)
    print("右侧相机矩阵：\n", cameraMatrix2)
    print("右侧畸变系数：\n", distCoeffs2)
    print("旋转矩阵：\n", R)
    print("平移向量：\n", T)
    print("本征矩阵：\n", E)
    print("基础矩阵：\n", F)
    print("rms：\n", retval)
    save_XML(retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F)
    # world_points,rotation_matrix, translation_vector,camera_matrix_left, dist_coeffs_left, imagepoints_left
    #read_XML()
    return objpoints, R, T, cameraMatrix1, distCoeffs1, rimgpoints

def calibrate_camera(image_folder_path, pattern_size, square_size, LR):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备棋盘格的世界坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # 存储棋盘格角点的世界坐标和图像坐标
    objpoints = []  # 世界坐标系中的三维点
    imgpoints = []  # 图像平面中的二维点
    for filename in sorted(os.listdir(image_folder_path)):
        print(filename)
        if LR == 1:
            if filename.__contains__("Right"):
                continue
        else:
            if filename.__contains__("Left"):
                continue
        file_path = os.path.join(image_folder_path, filename)
        img = cv2.imread(file_path)
        show_img = resize_image(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray = img.copy()
        # 查找棋盘格角点
        gray = resize_image(gray)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        print("found something",ret, corners)
        # 如果找到了棋盘格角点
        if ret == True:
            objpoints.append(objp)

            # 进行亚像素级角点检测
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # 绘制并显示棋盘格角点
            show_img = cv2.drawChessboardCorners(show_img, pattern_size, corners2, ret)
            cv2.imshow('img', show_img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # 进行相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("相机标定结果：")
    print("相机矩阵：\n", mtx)
    print("畸变系数：\n", dist)
    print("旋转向量：\n", rvecs)
    print("平移向量：\n", tvecs)
    return ret, mtx, dist, rvecs, tvecs, imgpoints
# 图像文件夹路径
image_folder_path = "./gray"

# 棋盘格尺寸（内角点数目）
pattern_size = (8, 11)

# 棋盘格方格尺寸（单位：毫米）
square_size = 10.0

# 调用函数进行相机标定
# left
# ret_left, mtx_left, dist_left, rvecs_left, tvecs_left, imgpoints_left = calibrate_camera(image_folder_path, pattern_size, square_size, 1)

# right
# ret_right, mtx_right, dist_right, rvecs_right, tvecs_right, imgpoints_right = calibrate_camera(image_folder_path, pattern_size, square_size, 0)

# cali!!!!!!
#world_points,rotation_matrix, translation_vector,camera_matrix_left, dist_coeffs_left, imagepoints_right = calibrate_camera_stereo(image_folder_path,pattern_size,square_size)

camera_matrix_left,dist_coeffs_left,camera_matrix_right,dist_coeffs_right,rotation_matrix,translation_vector,E, F =read_XML('stereo_calibration_data.xml')


# 计算矫正映射
image_size = (int(5472/8), int(3648/8))  # 图像大小，根据实际情况调整
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camera_matrix_left, dist_coeffs_left,
                                            camera_matrix_right, dist_coeffs_right,
                                            image_size, rotation_matrix, translation_vector)
map1_left, map2_left = cv2.initUndistortRectifyMap(camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)
# 加载左右图像
image_left = cv2.imread('./gray_Left9.bmp', 0)  # 转为灰度图像
image_right = cv2.imread('./gray_Right9.bmp', 0)  # 转为灰度图像
image_left,_ = resize_image(image_left)
image_right,_ = resize_image(image_right)

# 对左图像进行矫正
rectified_left = cv2.remap(image_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)

# 对右图像进行矫正
rectified_right = cv2.remap(image_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)

rectimage = cv2.hconcat([rectified_left, rectified_right])
rectimage = cv2.cvtColor(rectimage, cv2.COLOR_GRAY2BGR)
for i in range(10):
    line_y = i * int(rectimage.shape[0]/10)
    cv2.line(rectimage, (0, line_y), (rectimage.shape[1]-1, line_y), (0, 0, 255), 1)
cv2.imwrite("output.jpg",rectimage)
overlap = np.uint8((rectified_left > 0) & (rectified_right > 0))
# 仅显示重叠区域
result_left = cv2.bitwise_and(rectified_left, rectified_left, mask=overlap)
result_right = cv2.bitwise_and(rectified_right, rectified_left, mask=overlap)
#ol = cv2.hconcat([result_left, result_right])
cv2.imshow('Overlap', result_right)
img2 = rectified_left[overlap]
cv2.waitKey(5000)
#投影
# 1. 去畸变
undistorted_point = cv2.undistortPoints(np.array([[[2800.0, 2200.0]]]), camera_matrix_left, dist_coeffs_left, None, camera_matrix_left)

# 2. 从像素坐标到相机坐标
normalized_point = np.array([undistorted_point[0][0][0], undistorted_point[0][0][1], 1.0]).reshape(3, 1)

# 3. 从左相机坐标系到右相机坐标系
R = np.array(rotation_matrix).reshape(3, 3)
T = np.array(translation_vector).reshape(3, 1)
point_in_right_camera = np.dot(R, normalized_point) + T

# 4. 从3D点到2D投影
projected_point = np.dot(camera_matrix_right, point_in_right_camera)
projected_point /= projected_point[2, 0]

print(projected_point[:2])

point_left = np.array([100, 100])

# 去畸变
undistorted_point_left = cv2.undistortPoints(np.array([[point_left]], dtype=np.float32), camera_matrix_left, dist_coeffs_left, None, camera_matrix_left)

# 反投影到3D空间。这里我们假设深度为1，但你可以根据需要更改这个值
z = 1
X = z * (undistorted_point_left[0,0,0] - camera_matrix_left[0,2]) / camera_matrix_left[0,0]
Y = z * (undistorted_point_left[0,0,1] - camera_matrix_left[1,2]) / camera_matrix_left[1,1]
point_3D = np.array([[X, Y, z]])

# 将3D点从左摄像机坐标系转换到右摄像机坐标系
point_3D_right_cam = np.dot(rotation_matrix, point_3D.T) + translation_vector

# 使用右摄像机的内部参数矩阵将3D点投影到右图上
point_right = cv2.projectPoints(point_3D_right_cam.T, np.eye(3), np.zeros((3,1)), camera_matrix_right, dist_coeffs_right)[0]

print("Projected point in right image: ", point_right[0,0])

rms = reprojection_error(world_points,rotation_matrix, translation_vector,camera_matrix_left, dist_coeffs_left, imagepoints_right)
print(rms)

# 双目相机标定步骤
flags = cv2.CALIB_FIX_INTRINSIC  # 设置标定标志，固定内参
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)  # 可选的双目标定迭代终止条件




#LAB
def histogram_matching(source, template):
    # Convert the images from RGB to L*a*b* space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab)
    template_lab = cv2.cvtColor(template, cv2.COLOR_BGR2Lab)

    # Split the LAB images into L, A and B channels
    l_s, a_s, b_s = cv2.split(source_lab)
    l_t, a_t, b_t = cv2.split(template_lab)

    # Match the L channel histograms
    l_s_matched = cv2.equalizeHist(l_s, l_t)

    # Merge the adjusted L channel with the A and B channels from source image
    matched_lab = cv2.merge((l_s_matched, a_s, b_s))
    
    # Convert back to RGB space
    matched_img = cv2.cvtColor(matched_lab, cv2.COLOR_Lab2BGR)
    return matched_img

source = cv2.imread('source.jpg')
template = cv2.imread('template.jpg')

matched_img = histogram_matching(source, template)
cv2.imshow('Matched Image', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

def color_transfer(source, template):
    # Convert images from RGB to float32
    source = np.float32(source)
    template = np.float32(template)
    
    # Calculate means and standard deviations for each channel in source and template images
    mean_src = np.mean(source, axis=(0,1))
    std_src = np.std(source, axis=(0,1))
    
    mean_tpl = np.mean(template, axis=(0,1))
    std_tpl = np.std(template, axis=(0,1))
    
    # Adjust the source image using the formula: (pixel - mean_src) * (std_tpl/std_src) + mean_tpl
    adjusted = (source - mean_src) * (std_tpl / std_src) + mean_tpl
    
    # Clip the values to be in [0, 255] range and convert back to uint8
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted

source = cv2.imread('source.jpg')
template = cv2.imread('template.jpg')

adjusted_img = color_transfer(source, template)
cv2.imshow('Color Adjusted Image', adjusted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
