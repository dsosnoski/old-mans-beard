
import cv2
import json
import numpy as np
import os

BASE_PATH = '/home/dennis/projects/wcc/images'
IMAGE_EXTENSION = '.jp2'
JSON_EXTENSION = '.json'
SAMPLE_SIZE = 1600
TEST_SIZE = 800
TEST_IMAGES = 2
DATA_FILE = '/home/dennis/projects/wcc/datax.npy'
META_FILE = '/home/dennis/projects/wcc/datax.json'


def loadImage(base_path):
    print(f'loading {base_path}')
    img = cv2.imread(base_path + IMAGE_EXTENSION, cv2.IMREAD_COLOR)
    prob = np.zeros(img.shape[:-1], np.uint8)
    if os.path.exists(base_path + JSON_EXTENSION):
        with open(base_path + JSON_EXTENSION) as f:
            labels = json.load(f)
        shapes = labels['shapes']
        for shape in shapes:
            points = np.array([shape['points']]).astype('int32')
            cv2.fillPoly(prob, points, 255)
    return img, prob.astype(np.float16) / 255.0


def loadData(directory):
    file_names = [os.path.splitext(n)[0] for _, _, names in os.walk(directory) for n in names if n.endswith(".jp2")]
    images = []
    probs = []
    for f in file_names:
        img, mask = loadImage(os.path.join(directory, f))
        images.append(img)
        probs.append(mask)
    return file_names, np.array(images), np.array(probs)


def split_samples(images, probs, size):
    split_images = []
    split_probs = []
    assert images[0].shape[0] % size == 0
    assert images[0].shape[1] % size == 0
    for image, actual in zip(images, probs):
        for row in range(0, image.shape[0], size):
            for col in range(0, image.shape[1], size):
                split_images.append(image[row:row+size, col:col+size])
                split_probs.append(actual[row:row+size, col:col+size])
    return np.array(split_images), np.array(split_probs)


def summarize_samples(name, probs):
    counts = np.array(sorted((probs != 0).sum(axis=(1, 2))))
    max_count = counts.max()
    if max_count < 9:
        print(f'{name} has maximum of {max_count} positive pixels of {np.product(probs[0].shape)}')
    else:
        ranges = counts * 9 // max_count
        distrib = np.bincount(ranges)
        print(f'{name} has maximum of {max_count} positive pixels of {np.product(probs[0].shape)} with distribution {distrib}')


names, inputs, probs = loadData(BASE_PATH)
use_samples, use_probs = split_samples(inputs, probs, SAMPLE_SIZE)
use_names = names
summarize_samples('training/validation', use_probs)
# permuted = np.random.permutation(len(use_samples))
# validate_count = int(round(len(permuted) * VALIDATE_FRACTION))
# train_samples = use_samples[permuted[validate_count:]]
# train_probs = use_probs[permuted[validate_count:]]
# validate_samples = use_samples[permuted[:validate_count]]
# validate_probs = use_probs[permuted[:validate_count]]
# print(validate_count)

with open(DATA_FILE, 'wb') as f:
    np.save(f, use_samples)
    np.save(f, use_probs)
    np.save(f, inputs)
metadata = { 'names': names }
with open(META_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)





