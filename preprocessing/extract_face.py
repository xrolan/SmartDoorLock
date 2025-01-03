import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
import gc

class FaceDetector:

  def __init__(self):
    self.detector = MTCNN()

  def extract_face(self, filename, dump_path, required_size=(224,224)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)

    results = self.detector.detect_faces(pixels)

    if len(results) == 0:
      del image, pixels, results
      gc.collect()
      return
    
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)
    image.save(dump_path)

    del face, image, pixels, results
    gc.collect()
  
  # load images and extract faces for all images in a directory
  def load_and_dump_faces(self, directory, target_directory):
    for filename in os.listdir(directory):
      target_path = target_directory + filename
      path = directory + filename
      self.extract_face(path, target_path)

  def process_dataset(self, directory, dump_directory):
    
    for id, subdir in enumerate(os.listdir(directory)):
      print(f"Processing Data {id}: {subdir}")
      path = directory + subdir + '/'
      dump_path = dump_directory + subdir + '/'

      os.mkdir(dump_path)
      
      if not os.path.isdir(path):
        continue

      self.load_and_dump_faces(path, dump_path)
      print(f"{subdir} DONE")

  
  def edit_file(self, train_dir, val_dir, train_dump, val_dump):
    self.process_dataset(train_dir, train_dump)
    self.process_dataset(val_dir, val_dump)


if __name__ == "__main__":
  face_extract = FaceDetector()

  ##################################
  # For all labelled (no unknown)
  ##################################

  TRAIN_PATH = "./31facesdata/train/"
  VAL_PATH = "./31facesdata/val/"
  TRAIN_DUMP = "./dataset-extracted/train/"
  VAL_DUMP = "./dataset-extracted/val/"

  face_extract.edit_file(TRAIN_PATH, VAL_PATH, TRAIN_DUMP, VAL_DUMP)

