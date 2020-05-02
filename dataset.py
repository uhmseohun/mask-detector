import xmltodict, json
import cv2
import numpy as np

DATA_COUNT = 678
IMAGE_SIZE = (26, 26)

def process_object(_object, image):
  bound = _object['bndbox']
  label = _object['name']

  if (label == 'good'):
    label = [1, 0, 0]
  elif (label == 'bad'):
    label = [0, 1, 0]
  elif (label == 'none'):
    label = [0, 0, 1]

  (x1, y1, x2, y2) = tuple(map(lambda x: int(x), bound.values()))

  cropped = image[y1:y2, x1:x2]
  resized = cv2.resize(cropped, dsize=IMAGE_SIZE)
  resized = resized.reshape(*IMAGE_SIZE, 1)
  resized = resized / 255

  return (resized, label)

def process_data():
  x_data = []
  y_data = []

  for index in range(DATA_COUNT):
    with open(f'dataset/labels/{index}.xml') as file:
      data = file.read()
    data = xmltodict.parse(data)
    data = json.dumps(data)
    data = json.loads(data)

    image = cv2.imread(f'dataset/images/{index}.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    objects = data['annotation']['object']
    if (type(objects) == list):
      for _object in objects:
        (x, y) = process_object(_object, image)
        x_data.append(x)
        y_data.append(y)
    elif (type(objects) == dict):
      (x, y) = process_object(objects, image)
      x_data.append(x)
      y_data.append(y)

    if (index % 20 == 0):
      print(f'{index} / {DATA_COUNT} Loaded...')

  x_data = np.array(x_data)
  y_data = np.array(y_data)

  np.save('dataset/images', x_data)
  np.save('dataset/labels', y_data)

  return (x_data, y_data)

def load_data():
  x_data = np.load('dataset/images.npy')
  y_data = np.load('dataset/labels.npy')

  return (x_data, y_data)
