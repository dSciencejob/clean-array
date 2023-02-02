import sys
sys.path.insert(0,"./")
import os, sys, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import Normalize
from blazeface import BlazeFace, GenerateFace

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
face_detector = BlazeFace().to(gpu)
face_detector.load_weights("blazeface.pth")
face_detector.load_anchors("anchors.npy")
_ = face_detector.train(False)
frames_per_video = 14
verbose = True
num_frames = 17
input_size = (224,224)
target_size = (128,128)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)
UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'mp4'}

class Detect(models.resnet.ResNet):
    def __init__(self, training=True):
        super(Detect, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 23, 3], 
                                        groups=32, 
                                        width_per_group=8)

        self.fc = nn.Linear(2048, 1)
        
def resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size[0] // w
        w = size[0]
    else:
        w = w * size[1] // h
        h = size[1]

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized

def change_shape(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

def postprocess(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        insets=(0, 0)
        if insets[0] > 0:
            W = frame.shape[1]
            p = int(W * insets[0])
            frame = frame[:, p:-p, :]
        if insets[1] > 0:
            H = frame.shape[1]
            q = int(H * insets[1])
            frame = frame[q:-q, :, :]
        return frame    
    
def read_at_indice(path, capture, frame_idxs):
        try:
            frames = []
            idxs_read = []
            for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
                # Get the next frame, but don't decode if we're not using it.
                ret = capture.grab()
                if not ret:
                    break
                # Need to look at this frame?
                current = len(idxs_read)
                if frame_idx == frame_idxs[current]:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:                       
                        break
                    frame = postprocess(frame)
                    frames.append(frame)
                    idxs_read.append(frame_idx)
            if len(frames) > 0:
                return np.stack(frames), idxs_read            
            return None
        except:
            return None    
        
def capture(path, num_frames, jitter=0, seed=None):

        assert num_frames > 0

        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return None

        frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int32)
        if jitter > 0:
            np.random.seed(seed)
            jitter_offsets = np.random.randint(-jitter, jitter, len(frame_idxs))
            frame_idxs = np.clip(frame_idxs + jitter_offsets, 0, frame_count - 1)

        result = read_at_indice(path, capture, frame_idxs)
        capture.release()
        return result

def predict_file(video_path, batch_size):
        faces = face_extractor.process_file(video_path)
        face_extractor.final_face(faces)
        if len(faces) > 0:
            x = np.zeros((batch_size, input_size[0], input_size[1], 3), dtype=np.uint8)
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    resized_face = resize_image(face, input_size)
                    resized_face = change_shape(resized_face)
                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
            if n > 0:
                x = torch.tensor(x, device=gpu).float()
                x = x.permute((0, 3, 1, 2))
                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)
                with torch.no_grad():
                    y_pred = model(x)
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    return y_pred[:n].mean().item()

def fake_detect(filename, num_workers):
    y_pred = predict_file(filename, batch_size=frames_per_video)
    return y_pred
    
model = Detect().to(gpu)
_ = model.eval()
del checkpoint
video_read = lambda x: capture(x, num_frames=frames_per_video)
face_extractor = GenerateFace(video_read, face_detector,input_size,target_size)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(filename):
    global model, graph
    result = fake_detect(filename, num_workers=4)
           
    return result

