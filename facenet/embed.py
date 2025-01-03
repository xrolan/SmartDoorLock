import os
import numpy as np

from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Face_Embed:

  def __init__(self, model_dir):
    self.detector = MTCNN()
    self.model = load_model(model_dir)

  def extract_face(self, filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)

    results = self.detector.detect_faces(pixels)

    if len(results) == 0:
      return np.asarray([-1])
    
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array
  
  def get_embedding(self, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    sample = np.expand_dims(face, axis=0)
    yhat = self.model.predict(sample)
    return yhat[0]

  # load images and extract faces for all images in a directory
  def load_faces(self, directory):
    faces = list()
    for filename in os.listdir(directory):
      path = directory + filename
      face = self.extract_face(path)
      if face.shape == (160, 160, 3):
        face = self.get_embedding(face)
        faces.append(face)
    return faces

  def load_dataset(self, directory):
    X, y = list(), list()
    
    for id, subdir in enumerate(os.listdir(directory)):
      path = directory + subdir + '/'
      
      if not os.path.isdir(path):
        continue

      faces = self.load_faces(path)
      labels = [id for _ in range(len(faces))]

      print(f"class {subdir} has {len(labels)} data")

      X.extend(faces)
      y.extend(labels)
    return np.asarray(X), np.asarray(y)

  
  def get_np(self, train_dir, val_dir):
    X_train, y_train = self.load_dataset(train_dir)
    X_test, y_test = self.load_dataset(val_dir)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
  MODEL_PATH = "./model/facenet_keras.h5"
  face_embed = Face_Embed(MODEL_PATH)

  ##################################
  # For all labelled (no unknown)
  ##################################

  TRAIN_PATH = "./31facesdata/train/"
  VAL_PATH = "./31facesdata/val/"
  EMBEDDING_DUMP_PATH = "./embedding/faces-embedding.npz"

  print(TRAIN_PATH)
  print(VAL_PATH)

  X_train, X_test, y_train, y_test = face_embed.get_np(TRAIN_PATH, VAL_PATH)

  np.savez_compressed(EMBEDDING_DUMP_PATH, X_train, y_train, X_test, y_test)

  ##################################
  # For some unknown
  ##################################

  TRAIN_PATH = "./31facesdata_unknown/train/"
  VAL_PATH = "./31facesdata_unknown/val/"
  EMBEDDING_DUMP_PATH = "./embedding/faces-embedding-unknown.npz"

  print(TRAIN_PATH)
  print(VAL_PATH)

  X_train, X_test, y_train, y_test = face_embed.get_np(TRAIN_PATH, VAL_PATH)

  np.savez_compressed(EMBEDDING_DUMP_PATH, X_train, y_train, X_test, y_test)