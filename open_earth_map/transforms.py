import random
import numbers
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image


class ToTensor:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        msks = [(sample["mask"] == v) * 1 for v in self.classes]
        sample["mask"] = TF.to_tensor(np.stack(msks, axis=-1))
        sample["image"] = TF.to_tensor(sample["image"])
        return sample


class Rotate:
    def __init__(self, degrees=(-180, 180)):
        self.degrees = degrees

    def __call__(self, sample):
        angle = random.uniform(*self.degrees)

        img = TF.rotate(sample["image"], angle, InterpolationMode.BICUBIC)
        msk = TF.rotate(sample["mask"], angle, InterpolationMode.NEAREST)
        return {"image": img, "mask": msk}


class Crop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample["mask"].size

        if h > self.size[0] and w > self.size[1]:
            i = random.randrange(0, h - self.size[0])
            j = random.randrange(0, w - self.size[1])
            img = TF.crop(sample["image"], i, j, *self.size)
            msk = TF.crop(sample["mask"], i, j, *self.size)
        else:
            img = TF.resize(sample["image"], self.size, InterpolationMode.BICUBIC)
            msk = TF.resize(sample["mask"], self.size, InterpolationMode.NEAREST)

        return {"image": img, "mask": msk}


class Resize:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        img = TF.resize(sample["image"], self.size, InterpolationMode.BICUBIC)
        msk = TF.resize(sample["mask"], self.size, InterpolationMode.NEAREST)

        return {"image": img, "mask": msk}


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        img = self.color_jitter(sample["image"])
        msk = sample["mask"]
        return {"image": img, "mask": msk}


class GaussianNoise:
    def __init__(self, mean=0, std=0.01, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            # Convert PIL image to numpy array
            img = (
                np.array(sample["image"], dtype=np.float32) / 255.0
            )  # Normalize to [0, 1]
            # Generate noise with the same shape as the image
            noise = np.random.normal(self.mean, self.std, img.shape).astype(np.float32)
            # Add noise to the image
            img = np.clip(img + noise, 0, 1)  # Clip to ensure values stay in [0, 1]
            # Convert back to PIL image
            img = Image.fromarray((img * 255).astype(np.uint8))
        else:
            img = sample["image"]
        msk = sample["mask"]
        return {"image": img, "mask": msk}


class RandomErasing:
    def __init__(self, probability=0.5, area_range=(0.02, 0.4), min_aspect_ratio=0.3):
        self.probability = probability
        self.area_range = area_range
        self.min_aspect_ratio = min_aspect_ratio

    def __call__(self, sample):
        if random.random() < self.probability:
            # 将 PIL 图像转换为 Tensor
            img_tensor = TF.to_tensor(sample["image"])
            msk_tensor = TF.to_tensor(sample["mask"])

            # 获取随机擦除参数
            x, y, w, h = self.get_random_erasing_params(sample["image"])

            # 对图像应用擦除
            img_tensor = TF.erase(img_tensor, x, y, h, w, v=0)

            # 对掩码应用擦除，填充值为 255（表示忽略的类别）
            msk_tensor = TF.erase(msk_tensor, x, y, h, w, v=0)

            # 将 Tensor 转换回 PIL 图像
            img = TF.to_pil_image(img_tensor)
            msk = TF.to_pil_image(msk_tensor)
        else:
            img = sample["image"]
            msk = sample["mask"]
        return {"image": img, "mask": msk}

    def get_random_erasing_params(self, image):
        # 获取随机擦除参数: x, y, h, w
        width, height = image.size
        area = random.uniform(*self.area_range) * width * height
        aspect_ratio = random.uniform(self.min_aspect_ratio, 1 / self.min_aspect_ratio)

        h = int(np.round(np.sqrt(area * aspect_ratio)))
        w = int(np.round(np.sqrt(area / aspect_ratio)))

        if w > width:
            w = width
        if h > height:
            h = height

        x = random.randint(0, width - w)
        y = random.randint(0, height - h)

        return x, y, h, w


# class Mosaic:
#     def __init__(self, img_size, max_objects=50):
#         self.img_size = img_size
#         self.max_objects = max_objects  # Placeholder for future functionality

#     def __call__(self, samples):
#         # Ensure we have four distinct samples
#         if len(samples) != 4:
#             raise ValueError("Mosaic transformation requires four samples. now the length is: " + str(len(samples)))

#         # Get images and masks from samples
#         images = [sample["image"] for sample in samples]
#         masks = [sample["mask"] for sample in samples]

#         # Resize images and masks to half the target size
#         w, h = self.img_size
#         half_w, half_h = w // 2, h // 2
#         resized_images = [img.resize((half_w, half_h)) for img in images]
#         resized_masks = [
#             msk.resize((half_w, half_h), resample=Image.NEAREST) for msk in masks
#         ]

#         # Create a new mosaic image and mask
#         mosaic_img = Image.new("RGB", (w, h))
#         mosaic_mask = Image.new("L", (w, h))

#         # Positions to paste images and masks
#         positions = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]

#         for img, msk, pos in zip(resized_images, resized_masks, positions):
#             mosaic_img.paste(img, pos)
#             mosaic_mask.paste(msk, pos)

#         # Apply random rotation
#         angle = random.uniform(-180, 180)
#         rotated_img = TF.rotate(mosaic_img, angle)
#         rotated_mask = TF.rotate(mosaic_mask, angle, resample=Image.NEAREST)

#         # Apply random cropping
#         rotated_w, rotated_h = rotated_img.size
#         crop_params = self.get_random_crop_params(rotated_img)
#         cropped_img = TF.crop(rotated_img, *crop_params)
#         cropped_mask = TF.crop(rotated_mask, *crop_params)

#         # Ensure cropped image and mask are of the target size
#         if cropped_img.size != self.img_size:
#             cropped_img = cropped_img.resize(self.img_size)
#             cropped_mask = cropped_mask.resize(self.img_size, resample=Image.NEAREST)

#         # Convert to tensors
#         tensor_img = TF.to_tensor(cropped_img)
#         tensor_mask = TF.to_tensor(
#             cropped_mask
#         ).long()  # Convert to long for class labels

#         return {"image": tensor_img, "mask": tensor_mask}

#     def get_random_crop_params(self, img):
#         """Get parameters for random cropping"""
#         width, height = img.size
#         crop_size = self.img_size
#         max_x = width - crop_size[0]
#         max_y = height - crop_size[1]
#         x = random.randint(0, max_x) if max_x > 0 else 0
#         y = random.randint(0, max_y) if max_y > 0 else 0
#         return y, x, crop_size[0], crop_size[1]  # (top, left, height, width)
