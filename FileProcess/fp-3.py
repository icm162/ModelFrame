import pandas as pd
import cv2 as cv
import os


path = r"F:\DAT\Aachen-Heerlen Annotated Steel Microstructure Dataset\nature_scidata_heerlen_aachen_steel_morph.pickle"
to_path = r"C:\Users\icm162\Desktop"

contrast_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255),(255,255,0)]

df = pd.read_pickle(path)
df:pd.DataFrame

num = 1262

print(df.iloc[0,:])

filtered = df[df["image_url"] == df.iloc[num,0]]

img_path = os.path.join(*(path.split("\\")[:-1] + ["images", df.iloc[num, 0]]))
img_path = img_path.replace(":", ":\\")

poly_points = []
for no in range(len(filtered)): poly_points.append(filtered.iloc[no,2])

image = cv.imread(img_path)
cv.imshow("image", image)
cv.imwrite(os.path.join(to_path, "image.png"), image)
for i, lines in enumerate(poly_points):
    for j, point in enumerate(lines):
        cv.line(image, point, lines[j-len(lines)+1], contrast_colors[i%len(contrast_colors)], 2, cv.LINE_AA)
cv.imshow("labeled", image)
cv.imwrite(os.path.join(to_path, "label.png"), image)
cv.waitKey(0)
cv.destroyAllWindows()

