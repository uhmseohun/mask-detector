import os
import cv2

for (root, dirs, files) in os.walk('dataset/images'):
    for (index, file) in enumerate(files):
        file_path = os.path.join(root, file)
        (file, ext) = os.path.splitext(file)

        image = cv2.imread(file_path)
        cv2.imwrite(f'dataset/images/{index}.jpg', image)

        os.remove(file_path)

        os.rename(
            f'dataset/labels/{file}.xml',
            f'dataset/labels/{index}.xml'
        )
