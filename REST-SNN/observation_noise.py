from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageChops
import math
import random
import numpy as np

class DataAugmentation:
    def __init__(self):
        pass
 
    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")
 
    @staticmethod
    # def randomShift(image):
    def randomShift(image, shift_strength=0.2):

        xoff = np.random.randint(0, math.ceil(image.size[0]*shift_strength))
        yoff = np.random.randint(0, math.ceil(image.size[1]*shift_strength))
        # return image.offset(xoffset = random_xoffset, yoffset = random_yoffset)
        # return image.offset(random_xoffset)
        width, height = image.size
        c = ImageChops.offset(image,xoff,yoff)
        c.paste((0,0,0),(0,0,xoff,height))
        c.paste((0,0,0),(0,0,width,yoff))
        return c

    @staticmethod
    def randomRotation(image, rotation_max_angle=30,mode=Image.BICUBIC):

        random_angle = np.random.randint(1, rotation_max_angle)
        return image.rotate(random_angle, mode)
 
    @staticmethod
    def randomCrop(image,crop_strength=0.2):

        image_width = image.size[0]
        image_height = image.size[1]
        crop_image_width = math.ceil(image_width*crop_strength)
        crop_image_height = math.ceil(image_height*crop_strength)
        x = np.random.randint(0, image_width - crop_image_width)
        y = np.random.randint(0, image_height - crop_image_height) 
        random_region = (x, y, x + crop_image_width, y + crop_image_height)
        return image.crop(random_region)
 
    @staticmethod
    def randomColor(image, color_strength=30):

        random_factor = np.random.randint(0, color_strength) / 10. 
        color_image = ImageEnhance.Color(image).enhance(random_factor)  
        random_factor = np.random.randint(10, color_strength) / 10.  
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
        random_factor = np.random.randint(10, color_strength) / 10.  
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  
        random_factor = np.random.randint(0, color_strength) / 10.  
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor) 
 
    @staticmethod
    def randomGaussian(image, mean=0.2):

        mean, sigma = mean,1.

        def gaussianNoisy(im, mean=0.2, sigma=0.3):

            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        img = np.array(image)
        img.flags.writeable = True  
        width, height = img.shape[:2]
        # try:
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        # except:
        #     img = img
        return Image.fromarray(np.uint8(img))
 
    @staticmethod
    def randomCutout(image, length=20):

        image = np.array(image)
        image.flags.writeable = True

        h, w, c = image.shape
        mask = np.ones((h, w), np.float32)

        n_holes=np.random.randint(1,4)

        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - length // 2, 0, h))
            y2 = int(np.clip(y + length // 2, 0, h))
            x1 = int(np.clip(x - length // 2, 0, w))
            x2 = int(np.clip(x + length // 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, c, axis=2)
        image = image * mask
        return Image.fromarray(np.uint8(image))
 
    @staticmethod
    def randomErasing(image, region_strength=0.2):

        image = np.array(image)
        image.flags.writeable = True

        region_h = math.ceil(image.shape[0]*region_strength)
        region_w = math.ceil(image.shape[1]*region_strength)

        x1 = random.randint(0, image.shape[1] - region_w)
        y1 = random.randint(0, image.shape[0] - region_h)

        image[y1:y1+region_h, x1:x1+region_w, 0] = np.random.randint(0, 255, size=(region_h, region_w))
        image[y1:y1+region_h, x1:x1+region_w, 1] = np.random.randint(0, 255, size=(region_h, region_w))
        image[y1:y1+region_h, x1:x1+region_w, 2] = np.random.randint(0, 255, size=(region_h, region_w))

        return Image.fromarray(np.uint8(image))
    
    @staticmethod
    def saveImage(image, path):
        try:
            image.save(path)
        except:
            print('not save img: ', path)
            pass


funcMap = {
    "shift": DataAugmentation.randomShift, 
    "rotation": DataAugmentation.randomRotation,
    "color": DataAugmentation.randomColor,
    "cutout": DataAugmentation.randomCutout,
    "erasing": DataAugmentation.randomErasing,
    }

DEFAULT_PARAMS = {
    "shift": 0.3,
    "rotation": 30,
    "color": 20,
    "cutout": 20,
    "erasing": 0.3,
}

def apply_all_noise(rgb_obs, select_types=None):


    if select_types is None:
        select_types = ["shift", "rotation", "color", "cutout", "erasing"]
    

    if isinstance(rgb_obs, np.ndarray):
        image = Image.fromarray(rgb_obs)
    else:
        image = rgb_obs

    for func_name in select_types:
        if func_name in funcMap:
            value = DEFAULT_PARAMS.get(func_name, 0)
            image = funcMap[func_name](image, value)
        else:
            pass
    return np.array(image)