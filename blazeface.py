import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2

class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class BlazeFace(nn.Module):

    input_size = (128, 128)
    
    def __init__(self):
        super(BlazeFace, self).__init__()

        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(24, 24),
            BlazeBlock(24, 28),
            BlazeBlock(28, 32, stride=2),
            BlazeBlock(32, 36),
            BlazeBlock(36, 42),
            BlazeBlock(42, 48, stride=2),
            BlazeBlock(48, 56),
            BlazeBlock(56, 64),
            BlazeBlock(64, 72),
            BlazeBlock(72, 80),
            BlazeBlock(80, 88),
        )
        
        self.backbone2 = nn.Sequential(
            BlazeBlock(88, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
        )

        self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

        self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
        self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)
        
    def forward(self, x):

        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        
        b = x.shape[0]   

        x = self.backbone1(x)        
        h = self.backbone2(x)        
        
        c1 = self.classifier_8(x)     
        c1 = c1.permute(0, 2, 3, 1)     
        c1 = c1.reshape(b, -1, 1)    

        c2 = self.classifier_16(h)     
        c2 = c2.permute(0, 2, 3, 1)   
        c2 = c2.reshape(b, -1, 1)   

        c = torch.cat((c1, c2), dim=1)  

        r1 = self.regressor_8(x)      
        r1 = r1.permute(0, 2, 3, 1)  
        r1 = r1.reshape(b, -1, 16)    

        r2 = self.regressor_16(h)      
        r2 = r2.permute(0, 2, 3, 1)   
        r2 = r2.reshape(b, -1, 16)  

        r = torch.cat((r1, r2), dim=1) 
        return [r, c]

    def _device(self):
        return self.classifier_8.weight.device
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()        
    
    def load_anchors(self, path):
        self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=self._device())
        assert(self.anchors.ndimension() == 2)
        assert(self.anchors.shape[0] == self.num_anchors)
        assert(self.anchors.shape[1] == 4)

    def _preprocess(self, x):
        return x.float() / 127.5 - 1.0

    def predict_on_image(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def predict_on_batch(self, x, apply_nms=True):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == 128
        assert x.shape[3] == 128

        x = x.to(self._device())
        x = self._preprocess(x)

        with torch.no_grad():
            out = self.__call__(x)

        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        return self.nms(detections) if apply_nms else detections

    def nms(self, detections):
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 17), device=self._device())
            filtered_detections.append(faces)

        return filtered_detections
    
    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):

        assert raw_box_tensor.ndimension() == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert raw_score_tensor.ndimension() == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]
        
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)
        
        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

        mask = detection_scores >= self.min_score_thresh

        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):

        boxes = torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2. 
        boxes[..., 1] = x_center - w / 2. 
        boxes[..., 2] = y_center + h / 2. 
        boxes[..., 3] = x_center + w / 2. 

        for k in range(6):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections):

        if len(detections) == 0: return []

        output_detections = []

        remaining = torch.argsort(detections[:, 16], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections    

class GenerateFace:
       
    def __init__(self, video_read, face_detector, input_size,target_size):

        self.video_read = video_read
        self.face_detector = face_detector
        self.input_size = input_size
        self.target_size = target_size
        
    def process_files(self, input_dir, filenames, video_idxs):

        videos_read = []
        frames_read = []
        frames = []
        tiles = []
        resize_info = []

        for video_idx in video_idxs:
            
            filename = filenames[video_idx]
            video_path = os.path.join(input_dir, filename)
            result = self.video_read(video_path)
            if result is None: continue
            videos_read.append(video_idx)
            my_frames, my_idxs = result
            frames.append(my_frames)
            frames_read.append(my_idxs)
            my_tiles, my_resize_info = self.detect_frames(my_frames)
            tiles.append(my_tiles)
            resize_info.append(my_resize_info)
        batch = np.concatenate(tiles)
        all_detections = self.face_detector.predict_on_batch(batch, apply_nms=False)
        result = []
        offs = 0
        
        for v in range(len(tiles)):            
            num_tiles = tiles[v].shape[0]
            detections = all_detections[offs:offs + num_tiles]
            offs += num_tiles
            detections = self.resize_frames(detections, resize_info[v])
            num_frames = frames[v].shape[0]
            frame_size = (frames[v].shape[2], frames[v].shape[1])
            detections = self.original_frames(num_frames, frame_size, detections)
            detections = self.face_detector.nms(detections)
            
            for i in range(len(detections)):
                faces = self.margin(detections[i], frame_size, 0.2)
                faces = self.crop(frames[v][i], faces)

                scores = list(detections[i][:, 16].cpu().numpy())
                frame_dict = { "video_idx": videos_read[v],
                               "frame_idx": frames_read[v][i],
                               "frame_w": frame_size[0],
                               "frame_h": frame_size[1],
                               "faces": faces, 
                               "scores": scores }
                result.append(frame_dict)
        return result

    def process_file(self, video_path):

        input_dir = os.path.dirname(video_path)
        filenames = [ os.path.basename(video_path) ]
        return self.process_files(input_dir, filenames, [0])

    def detect_frames(self, frames):

        num_frames, H, W, _ = frames.shape
        split_size = min(H, W)
        x_step = (W - split_size) // 2
        y_step = (H - split_size) // 2
        num_v = 1
        num_h = 3 if W > H else 1
        splits = np.zeros((num_frames * num_v * num_h, self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        i = 0
        for f in range(num_frames):
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    crop = frames[f, y:y+split_size, x:x+split_size, :]
                    splits[i] = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_AREA)
                    x += x_step
                    i += 1
                y += y_step

        resize_info = [split_size / self.target_size[0], split_size / self.target_size[1], 0, 0]
        return splits, resize_info

    def resize_frames(self, detections, resize_info):
        projected = []
        target_w, target_h = self.target_size
        scale_w, scale_h, offset_x, offset_y = resize_info
        for i in range(len(detections)):
            detection = detections[i].clone()
            for k in range(2):
                detection[:, k*2    ] = (detection[:, k*2    ] * target_h - offset_y) * scale_h
                detection[:, k*2 + 1] = (detection[:, k*2 + 1] * target_w - offset_x) * scale_w
            for k in range(2, 8):
                detection[:, k*2    ] = (detection[:, k*2    ] * target_w - offset_x) * scale_w
                detection[:, k*2 + 1] = (detection[:, k*2 + 1] * target_h - offset_y) * scale_h
            projected.append(detection)
        return projected    
    
    def original_frames(self, num_frames, frame_size, detections):

        combined_detections = []

        W, H = frame_size
        split_size = min(H, W)
        x_step = (W - split_size) // 2
        y_step = (H - split_size) // 2
        num_v = 1
        num_h = 3 if W > H else 1
        i = 0
        for f in range(num_frames):
            detections_for_frame = []
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    detection = detections[i].clone()
                    if detection.shape[0] > 0:
                        for k in range(2):
                            detection[:, k*2    ] += y
                            detection[:, k*2 + 1] += x
                        for k in range(2, 8):
                            detection[:, k*2    ] += x
                            detection[:, k*2 + 1] += y

                    detections_for_frame.append(detection)
                    x += x_step
                    i += 1
                y += y_step
            combined_detections.append(torch.cat(detections_for_frame))
        return combined_detections
    
    def margin(self, detections, frame_size, margin=0.2):

        offset = torch.round(margin * (detections[:, 2] - detections[:, 0]))
        detections = detections.clone()
        detections[:, 0] = torch.clamp(detections[:, 0] - offset*2, min=0)            
        detections[:, 1] = torch.clamp(detections[:, 1] - offset, min=0)              
        detections[:, 2] = torch.clamp(detections[:, 2] + offset, max=frame_size[1])  
        detections[:, 3] = torch.clamp(detections[:, 3] + offset, max=frame_size[0])  
        return detections
    
    def crop(self, frame, detections):

        faces = []
        for i in range(len(detections)):
            ymin, xmin, ymax, xmax = detections[i, :4].cpu().numpy().astype(np.int32)
            face = frame[ymin:ymax, xmin:xmax, :]
            faces.append(face)
        return faces

    def remove_crops(self, crops, pct=0.1):

        for i in range(len(crops)):
            frame_data = crops[i]
            video_area = frame_data["frame_w"] * frame_data["frame_h"]
            faces = frame_data["faces"]
            scores = frame_data["scores"]
            new_faces = []
            new_scores = []
            for j in range(len(faces)):
                face = faces[j]
                face_H, face_W, _ = face.shape
                face_area = face_H * face_W
                if face_area / video_area < 0.1:
                    new_faces.append(face)
                    new_scores.append(scores[j])
            frame_data["faces"] = new_faces
            frame_data["scores"] = new_scores

    def final_face(self, crops):

        for i in range(len(crops)):
            frame_data = crops[i]
            if len(frame_data["faces"]) > 0:
                frame_data["faces"] = frame_data["faces"][:1]
                frame_data["scores"] = frame_data["scores"][:1]


def intersect(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):

    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)
