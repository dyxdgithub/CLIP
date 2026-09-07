import os
import cv2
import numpy as np


def generate_hard_negative_via_fft(
    anchor_path, negative_path, output_path, r_threshold=0.1
):
    """通过傅里叶变换（FFT）融合锚点（正样本）低频与负样本高频，生成新的硬负样本。

    参数:
        anchor_path (str): 正样本（锚点）图像路径
        negative_path (str): 负样本图像路径
        output_path (str): 生成的新负样本保存路径
        r_threshold (float): 中心低频掩码的半径比例阈值，默认 0.1
    """
    # 1. 读取图像并转换为单通道灰度图（或浮点型数据）
    # 注意：FFT 针对 2D 矩阵处理，若为彩色图可逐通道处理，这里以单通道灰度图为例
    img_anchor = cv2.imread(anchor_path, cv2.IMREAD_GRAYSCALE)
    img_negative = cv2.imread(negative_path, cv2.IMREAD_GRAYSCALE)

    if img_anchor is None or img_negative is None:
        raise ValueError("图像读取失败，请检查图像路径是否正确！")

    # 确保两张图尺寸一致（若不一致，将负样本 resize 到与正样本相同尺寸）
    h, w = img_anchor.shape
    if img_negative.shape != (h, w):
        img_negative = cv2.resize(
            img_negative, (w, h), interpolation=cv2.INTER_LINEAR
        )

    # 2. 对两张图分别进行二维快速傅里叶变换 (FFT)
    # np.fft.fft2 得到频域复数矩阵
    fft_anchor = np.fft.fft2(img_anchor)
    fft_negative = np.fft.fft2(img_negative)

    # 将频域的零频（低频中心）平移到频谱矩阵的中心
    fft_anchor_shift = np.fft.fftshift(fft_anchor)
    fft_negative_shift = np.fft.fftshift(fft_negative)

    # 提取各自的幅度谱 (Magnitude) 和 相位谱 (Phase)
    mag_anchor = np.abs(fft_anchor_shift)
    phase_anchor = np.angle(fft_anchor_shift)

    mag_negative = np.abs(fft_negative_shift)
    phase_negative = np.angle(fft_negative_shift)

    # 3. 生成保留中心低频区域的二值掩码 (Mask)
    # 中心点坐标 (cy, cx)
    cy, cx = h // 2, w // 2
    # 按照图像最小边长的比例设定低频半径
    max_radius = min(h, w) / 2
    r_cutoff = max_radius * r_threshold

    # 创建网格坐标矩阵计算每个点到中心的距离
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # 低频区域（距离中心 <= r_cutoff）设为 1，高频区域设为 0
    mask = np.where(dist_from_center <= r_cutoff, 1.0, 0.0)

    # 4. 利用掩码将正样本低频幅度与负样本高频幅度进行融合
    # mask * mag_anchor 保留正样本低频； (1 - mask) * mag_negative 保留负样本高频
    mag_fused = mask * mag_anchor + (1.0 - mask) * mag_negative

    # 5. 将合成后的幅度谱与负样本原始的相位谱组合
    # 复数公式：F = Magnitude * exp(1j * Phase)
    fft_fused_shift = mag_fused * np.exp(1j * phase_negative)

    # 将零频点逆平移回左上角
    fft_fused = np.fft.ifftshift(fft_fused_shift)

    # 通过逆快速傅里叶变换 (iFFT) 映射回图像空间
    img_fused_complex = np.fft.ifft2(fft_fused)

    # 取实部并限制像素范围在 [0, 255] 内
    img_fused = np.real(img_fused_complex)
    img_fused = np.clip(img_fused, 0, 255).astype(np.uint8)

    # 保存生成的难负样本图像
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(output_path, img_fused)
    print(f"成功生成新的难负样本并保存至: {output_path}")

    return img_fused


# ================= 使用示例 =================
if __name__ == "__main__":
    # 替换为你的本地图片路径
    anchor_img = "path/to/anchor_pos.jpg"
    negative_img = "path/to/hard_neg.jpg"
    save_img = "path/to/output_fused_negative.jpg"

    # 执行融合生成
    # generate_hard_negative_via_fft(anchor_img, negative_img, save_img, r_threshold=0.1)