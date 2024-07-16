import cv2
import csv
import os
import dlib
import sys
import numpy as np
from pathlib import Path
from skimage import transform as trans
from imutils import face_utils

def write_to_csv(img_names, base_dataset: Path):
    with open(str(base_dataset), 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['filename', 'label', 'targeted_label'])
        
        for img_name in img_names:
            writer.writerow([img_name, 1, 0]) 

def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark, mask_face
    
    else:
        return None, None, None

def video_manipulate(
    movie_path: Path,
    dataset_path: Path,
    num_frames: int, 
    stride: int, 
    ) -> None:
    """
    Processes a single video file by detecting and cropping the largest face in each frame and saving the results.

    Args:
        movie_path (str): Path to the video file to process.
        dataset_path (str): Path to the dataset directory.
        mask_path (str): Path to the mask directory.
        num_frames (int): Number of frames to extract from the video.
        stride (int): Number of frames to skip between each frame extracted.

    Returns:
        None
    """

    # Define face detector and predictor models
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './dlib_tools/shape_predictor_81_face_landmarks.dat'
    ## Check if predictor path exists
    if not os.path.exists(predictor_path):
        print(f"Predictor path does not exist: {predictor_path}")
        sys.exit()

    face_predictor = dlib.shape_predictor(predictor_path)
    
    def facecrop(
        org_path: Path,
        save_path: Path, 
        num_frames: int, 
        stride: int,
        face_predictor: dlib.shape_predictor, 
        face_detector: dlib.fhog_object_detector,
        ) -> None:
        """
        Helper function for cropping face and extracting landmarks.
        """
        
        # Open the video file
        assert org_path.exists(), f"Video file {org_path} does not exist."
        cap_org = cv2.VideoCapture(str(org_path))
        if not cap_org.isOpened():
            print(f"Failed to open {org_path}")
            return
        
        # Get the number of frames in the video
        frame_count_org = min(int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES) 

        # Get the frame rate of the video by dividing the number of frames by the duration (same interval between frames)
        # frame_idxs = np.arange(0, frame_count_org, stride, dtype=int)

        # Iterate through the frames
        file_names = []
        for cnt_frame in range(frame_count_org):
            ret_org, frame_org = cap_org.read()

            height, width = frame_org.shape[:-1]

            # Check if the frame was successfully read
            if not ret_org:
                print(f"Failed to read frame {cnt_frame} of {org_path}")
                break
            
            '''# Check if the frame is one of the frames to extract
            if cnt_frame not in frame_idxs:
                continue'''

            # Use the function to extract the aligned and cropped face
            cropped_face, landmarks, _ = extract_aligned_face_dlib(face_detector, face_predictor, frame_org)
            
            # Check if a face was detected and cropped
            if cropped_face is None:
                print(f"No faces in frame {cnt_frame} of {org_path}")
                continue
            
            # Check if the landmarks were detected
            if landmarks is None:
                print(f"No landmarks in frame {cnt_frame} of {org_path}")
                continue

            # Save cropped face, landmarks, and visualization image
            save_path_ = save_path / 'frames_aug' / org_path.stem
            save_path_.mkdir(parents=True, exist_ok=True)

            # Save cropped face
            image_path_ = save_path_ / 'images'
            image_path_.mkdir(parents=True, exist_ok=True)
            image_path = save_path_ / 'images' / f"{cnt_frame:03d}.png"
            
            if not image_path.is_file():
                cv2.imwrite(str(image_path), cropped_face)
            file_names.append(f"{cnt_frame:03d}.png")

            # Save landmarks
            #land_path = save_path / 'landmarks_aug' / org_path.stem / f"{cnt_frame:03d}.npy"
            #os.makedirs(os.path.dirname(land_path), exist_ok=True)
            #np.save(str(land_path), landmarks)

        # Release the video capture
        cap_org.release()
        write_to_csv(file_names, save_path_ / 'labels.csv')

    # Iterate through the videos in the dataset and extract faces
    try:
        facecrop(movie_path, dataset_path, num_frames, stride, face_predictor, face_detector)
    except Exception as e:
        print(e)
    
# Cap each video length to 100 frames   
MAX_FRAMES = 100

if __name__=="__main__": 
    video_directory = sys.argv[1]
    video_name = sys.argv[2]
    video_manipulate(Path(os.path.join(video_directory, video_name)), Path(video_directory), 1, 1)
