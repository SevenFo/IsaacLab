from typing import Tuple,List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def visualize_all(
    image: np.ndarray,
    bboxes: Optional[List[List[int]]] = None,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    dpi: int = 100,
    return_result: bool = False,
) -> None:
    """
    可视化图像、边界框和mask

    参数:
    image: RGB格式的输入图像 [H, W, 3]
    bboxes: 边界框列表 [[x1, y1, x2, y2], ...]
    mask: 多目标mask数组，每个像素值为对象ID (0表示背景)
    save_path: 图片保存路径 (None则显示)
    dpi: 输出图像分辨率
    """
    # 创建画布
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Visualization Results", fontsize=16)
    axes = axs.flatten()

    # 子图1: 原始图像
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 子图2: 带边界框的图像
    axes[1].imshow(image)
    if bboxes:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            axes[1].add_patch(rect)
    axes[1].set_title("With Bounding Boxes")
    axes[1].axis("off")

    # 子图3: 单独mask
    if mask is not None:
        # 生成彩色mask (忽略0背景)
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        unique_ids = np.unique(mask)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            color = plt.cm.get_cmap("tab10")(obj_id % 10)[
                :3
            ]  # 使用tab10颜色循环
            colored_mask[mask == obj_id] = np.array(color) * 255
        axes[2].imshow(colored_mask)
        axes[2].set_title("Segmentation Mask")
    else:
        axes[2].imshow(np.zeros_like(image))
        axes[2].set_title("No Mask Available")
    axes[2].axis("off")

    # 子图4: 叠加mask的图像
    axes[3].imshow(image)
    overlay = None
    if mask is not None:
        # 创建半透明覆盖层
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            color = plt.cm.get_cmap("tab10")(obj_id % 10)
            overlay[mask == obj_id] = [*color[:3], 0.5]  # RGBA格式

        axes[3].imshow(overlay)
    axes[3].set_title("Image with Mask Overlay")
    axes[3].axis("off")

    # 调整布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()
    if return_result:
        return overlay
