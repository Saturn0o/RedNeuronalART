from PIL import Image, ImageOps
import numpy as np

def process_image(image_path, size=(28, 28)):
    try:
        img = Image.open(image_path).convert('L')
        img = ImageOps.invert(img)
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        img_bw = img_resized.point(lambda x: 255 if x > 128 else 0, '1')
        data = np.array(img_bw.getdata(), dtype=np.uint8)
        pattern = 1 - (data / 255).astype(np.int8)
        return pattern.flatten(), img_bw

    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        return None, None