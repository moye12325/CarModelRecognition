import os

import cv2
from PIL import ImageEnhance, Image

filename = 'baojun.jpg'

# 图像几何增强
def geometric_augmentation(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 图像翻转（水平翻转）
    flipped_image = cv2.flip(image, 1)

    # 图像缩放
    scale_percent = 150  # 调整缩放比例
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # 图像旋转
    rotation_angle = 30  # 顺时针旋转30度
    (h, w) = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # 保存增强后的图像
    cv2.imwrite(output_path + filename + 'flipped_image.jpg', flipped_image)
    cv2.imwrite(output_path + filename + 'resized_image.jpg', resized_image)
    cv2.imwrite(output_path + filename + 'rotated_image.jpg', rotated_image)


# 图像颜色增强
def color_augmentation(image_path, output_path):
    # 读取图像
    image = Image.open(image_path)

    # 调整亮度
    brightness_enhancer = ImageEnhance.Brightness(image)
    enhanced_brightness = brightness_enhancer.enhance(1.5)  # 调整亮度增强度
    enhanced_brightness.save(output_path + filename + 'enhanced_brightness.jpg')

    # 调整对比度
    contrast_enhancer = ImageEnhance.Contrast(image)
    enhanced_contrast = contrast_enhancer.enhance(1.5)  # 调整对比度增强度
    enhanced_contrast.save(output_path + filename + 'enhanced_contrast.jpg')

    # 调整饱和度
    saturation_enhancer = ImageEnhance.Color(image)
    enhanced_saturation = saturation_enhancer.enhance(1.5)  # 调整饱和度增强度
    enhanced_saturation.save(output_path + filename + 'enhanced_saturation.jpg')


# 示例调用
# filename = 'baojun.jpg'
image_path = os.path.join('.', 'car_image', filename)
print(image_path)

output_path = './output/'  # 增强图像输出路径

geometric_augmentation(image_path, output_path)
color_augmentation(image_path, output_path)
