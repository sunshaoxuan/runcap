import numpy as np
from PIL import Image
import argparse

def compareimg(path1, path2, output_path):
    # 加载图片并转换为灰度
    img1 = Image.open(path1).convert('L')
    img2 = Image.open(path2).convert('L')

    # 将图片转为Numpy数组
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # 计算两个数组的差异，并确定哪些像素点是不同的
    difference = np.abs(arr1.astype(int) - arr2.astype(int))
    diff = (difference > 0).astype(np.uint8)  # 将差异转化为0或1

    # 创建输出图片
    output_image = np.zeros((arr1.shape[0], arr1.shape[1], 3), dtype=np.uint8)

    # 对于灰度部分，使用原图的灰度值
    output_image[:, :, 0] = arr1 * (1 - diff)
    output_image[:, :, 1] = arr1 * (1 - diff) + 255 * diff  # 差异处标记为绿色
    output_image[:, :, 2] = arr1 * (1 - diff)

    # 将Numpy数组转回图片
    output_img = Image.fromarray(output_image, 'RGB')
    output_img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Compare two images and highlight differences.")
    parser.add_argument("path1", type=str, help="Path to the first image")
    parser.add_argument("path2", type=str, help="Path to the second image")
    parser.add_argument("output_path", type=str, help="Path for the output difference image")

    args = parser.parse_args()

    compareimg(args.path1, args.path2, args.output_path)

if __name__ == "__main__":
    main()
