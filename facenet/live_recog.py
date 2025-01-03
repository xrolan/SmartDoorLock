import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from keras.models import load_model
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def load_facenet_model(model_path):
    model = load_model(model_path)
    return model

def preprocess_face(face_img, required_size=(160, 160)):
    face_img = cv2.resize(face_img, required_size)
    face_img = face_img.astype('float32')
    mean, std = face_img.mean(), face_img.std()
    face_img = (face_img - mean) / std
    face_img = np.expand_dims(face_img, axis=0)
    return face_img


def get_embedding(face, model):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    sample = np.expand_dims(face, axis=0)
    yhat = model.predict(sample)
    return yhat[0]

def load_known_faces():
  faces_npz = np.load("./embedding/faces-embedding.npz")

  X_train, y_train, X_test, y_test = faces_npz["arr_0"], faces_npz["arr_1"], faces_npz["arr_2"], faces_npz["arr_3"]
  X = np.concatenate((X_train, X_test))
  y = np.concatenate((y_train, y_test))
  
  return X, y


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)


def recognize_face(embedding, embd, labels, threshold=0.65):
  name = "Unknown"
  min_dist = float("-inf")
  for embed, label in zip(embd, labels):
    dist = cosine_similarity(embedding, embed)
    if dist > min_dist:
        if dist > threshold:
          name = label
        min_dist = dist
  return name, min_dist

def real_time_face_recognition(model, emb, label):
  cap = cv2.VideoCapture(0)
  detector = MTCNN()
  required_size = (160,160)

  while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)

    for face in faces:
      pixels = np.asarray(frame)

      x1, y1, width, height = face['box']
      x1, y1 = abs(x1), abs(y1)
      x2, y2 = x1 + width, y1 + height

      face = pixels[y1:y2, x1:x2]

      image = Image.fromarray(face)
      image = image.resize(required_size)
      face_img = np.asarray(image)

      embedding = get_embedding(face_img, model)

      name, dist = recognize_face(embedding, emb, label)

      text = name + " (" + str(dist) + ")"

      if name != "Unknown":
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
      else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_facenet_model("./model/facenet_keras.h5")
    X, y = load_known_faces()

    real_time_face_recognition(model, X, y)