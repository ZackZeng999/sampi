import torch
import numpy as np
from PIL import Image, ImageDraw, ImageColor
#################################### For Image ####################################
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam3_image_model(checkpoint_path="/root/autodl-tmp/sam3_model/sam3.pt")
processor = Sam3Processor(model)
# Load an image
image = Image.open("/root/proj/shouji.jpg")
with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
    inference_state = processor.set_image(image)
# Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="a mobile phone")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

boxes_np = boxes.cpu().numpy()
masks_np = masks.cpu().numpy()
if masks_np.shape[1] == 1:
    masks_np = masks_np.squeeze(1) # 移除第 1 维 (索引为 1)，形状变为 (N, H, W)

# 创建用于绘制 box 和 mask 的图像副本
image_for_boxes = image.copy()
image_for_masks = image.copy()

# 创建绘图对象
draw_boxes = ImageDraw.Draw(image_for_boxes)
draw_masks = ImageDraw.Draw(image_for_masks)

# 为每个实例生成一个唯一的颜色
colors = [
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan", "magenta",
    "lime", "teal", "olive", "maroon", "navy", "aqua", "fuchsia", "silver", "gray",
    "gold", "indigo"
]

# 设置绘制参数
box_width = 2      # 边界框线条宽度
mask_alpha = 0.5   # 掩码的透明度 (0.0 完全透明, 1.0 完全不透明)

# 遍历每个实例进行绘制
for i, (box, mask) in enumerate(zip(boxes_np, masks_np)):
    # 获取当前实例的颜色
    color_name = colors[i % len(colors)]
    color_rgb = ImageColor.getrgb(color_name)

    # 1. 绘制边界框 (Box)
    x_min, y_min, x_max, y_max = box
    draw_boxes.rectangle([x_min, y_min, x_max, y_max], outline=color_rgb, width=box_width)

    # 2. 绘制掩码 (Mask)
    # 创建一个与原图大小相同的、具有透明通道的图像层
    mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)

    # 将 mask 转换为 PIL Image 对象 (1-bit 或 8-bit)
    # 确保 mask 是 (H, W) 形状，并且是 0-255 的整数
    # SAM 的 mask 是布尔值，需要先转成 0-255 的整数
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8, 'L') # 使用 'L' 模式创建灰度图

    # 将 mask 作为蒙版，绘制一个彩色的矩形区域
    # 这里我们用一个半透明的纯色填充整个 mask 区域
    colored_mask = Image.new("RGBA", image.size, (*color_rgb, int(255 * mask_alpha)))
    mask_layer.paste(colored_mask, (0, 0), mask_pil)

    # 将 mask 层叠加到原图上
    image_for_masks = Image.alpha_composite(image_for_masks.convert("RGBA"), mask_layer).convert("RGB")

# 生成输出图片路径
image_path = "/root/proj/shouji.jpg"
base_path = image_path.rsplit('.', 1)
output_boxes_path = f"{base_path[0]}_with_boxes.{base_path[1]}"
output_masks_path = f"{base_path[0]}_with_masks.{base_path[1]}"

# 保存图片
image_for_boxes.save(output_boxes_path)
image_for_masks.save(output_masks_path)

print(f"\n边界框绘制结果已保存到: {output_boxes_path}")
print(f"掩码绘制结果已保存到: {output_masks_path}")