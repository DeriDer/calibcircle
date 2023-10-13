import os
import cv2

def convert_images_to_grayscale(folder_path, save_folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查是否为图像文件
        if os.path.isfile(file_path) and any(file_path.lower().endswith(extension) for extension in ['.bmp', '.jpeg', '.png']):
            # 读取图像
            image = cv2.imread(file_path)
            print(file_path)
            # 将图像转换为灰度图
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 构造保存路径
            save_path = os.path.join(save_folder_path, f"gray_{filename}")

            # 保存灰度图像
            cv2.imwrite(save_path, grayscale_image)

            print("已保存灰度图像：", save_path)

os.makedirs('gray', exist_ok=True)
convert_images_to_grayscale('dst','gray')