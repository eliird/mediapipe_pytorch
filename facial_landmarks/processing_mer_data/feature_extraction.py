import os
import cv2
import numpy as np
from numpy import ndarray
import mediapipe as mp
from typing import Tuple, Any
import random
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt


MODEL_PATH =  '/home/cv/workspace/eliird/mediapipe_pytorch/facial_landmarks/model_weights/face_landmarker.task'
MIN_CONFIDENCE = 0.5

# MEDIAPIPE Initialization
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    min_face_detection_confidence = MIN_CONFIDENCE,
    min_face_presence_confidence = MIN_CONFIDENCE,
)

# CREATE A CONSTANT LANDMARKER THAT EXISTS TILL APPLICATION CLOSES
LANDMARKER = FaceLandmarker.create_from_options(options)


def get_mp_detecion(image: Tuple[str, ndarray]) -> Any:
    if isinstance(image, str):
        if not os.path.exists(image):
            raise ValueError("Invalid file path")
        image = cv2.imread(image)
        
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection = LANDMARKER.detect(mp_image)
    return detection


def get_landmarks(image: Tuple[str, ndarray]) -> Tuple[ndarray, None]:
    
    face_landmarker_result = get_mp_detecion(image)
    
    try:
        landmarks = [
            [landmark.x, landmark.y, landmark.z]
            for landmark in face_landmarker_result.face_landmarks[0]]
    except:
        return None
    
    return np.array(landmarks)


def save_landmarks(image: Tuple[str, ndarray], save_path: str) -> bool:
    succeed = False
    
    landmarks = get_landmarks(image)
    if landmarks is not None:
        np.save(save_path, landmarks)
        succeed = True
                
    return succeed


def save_image(image: ndarray, save_path: str):
    if '.jpg' not in save_path:
        save_path += '.jpg'
    cv2.imwrite(save_path, image)
    return


def process_video(path: str, emotion: str, save_folder: str,num_imgs=10):
    
    if not os.path.exists(path):
        raise ValueError("Invalid Video Path")
    
    save_folder = os.path.join(save_folder, emotion)
    filename = path.split('/')[-1].split('.')[0]
    
    cap = cv2.VideoCapture(path)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count > num_imgs:
        frames_to_save = random.sample(range(0, frame_count + 1), num_imgs)
    else:
        frames_to_save = [*range(0, frame_count)]
        
    count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        count += 1
        if count -1 not in frames_to_save:
            continue
        
        save_path = os.path.join(save_folder, 'landmarks', filename + f'_{saved_count}.npy') 
        saved = save_landmarks(frame, save_path=save_path)
        if saved:
            save_path = os.path.join(save_folder, 'images', filename + f'_{saved_count}.jpg') 
            save_image(frame, save_path)
        
        saved_count += 1
        if saved_count == num_imgs:
            break
        
    return



def process_video_all_frames(path: str, emotion: str, save_folder: str, num_imgs=10):
    '''

    '''
    if not os.path.exists(path):
        raise ValueError("Invalid Video Path")
    
    save_folder = os.path.join(save_folder, emotion)
    filename = path.split('/')[-1].split('.')[0]
    
    cap = cv2.VideoCapture(path)
            
    temporal_landmarks = []
    while True:
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        landmarks = get_landmarks(frame)
        if landmarks is None:
            continue
        temporal_landmarks.append(landmarks)
        

    for i in range(len(temporal_landmarks), num_imgs):
        temporal_landmarks.append(temporal_landmarks[-1])
        
        
    save_path = os.path.join(save_folder, 'landmarks_temporal', filename + '.npy') 
    np.save(save_path, np.array(temporal_landmarks))
    return


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()