from PIL import Image
import numpy as np

img = Image.open("F:\\all\salt\\train\images\\000e218f21.png")

img = img.convert('L')

mat = np.array(img)

print mat