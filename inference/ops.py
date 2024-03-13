import cv2
import numpy as np


class Compose:
    def __init__(self, op):
        self.op = op
    def __call__(self, im):
        for op in self.op:
            im = op(im)
        return im
    
class Resize:
    def __init__(self, size=(640, 640), letterbox=True, color=(114, 114, 114), is_rgb=False):
        if isinstance(size, int):
            size = (size, size)
        self.size = size # (w, h)
        self.letterbox = letterbox
        self.is_rgb = is_rgb
        self.color = color
    
    def __call__(self, im):
        if not self.is_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
        if not self.letterbox:
            return cv2.resize(im, dsize=self.size)
        
        shape = im.shape[:2] # (h, w)
        r = min(self.size[0] / shape[1], self.size[1] / shape[0])
        scale_shape = (int(shape[1] * r), int(shape[0] * r)) # (w, h)

        pad = (self.size[0] - scale_shape[0], self.size[1] - scale_shape[1])
        pad_w = pad[0] / 2
        pad_h = pad[1] / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        
        im = cv2.resize(im, dsize=scale_shape)
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        
        return im

class ToTensor:
    def __call__(self, im):
        im = np.transpose(im, (2, 0, 1))
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32) / 255.0
        return im
        
class Normalize:

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, im):
        mean = self.mean[:, None, None]
        std = self.std[:, None, None]
        return (im - mean) / std

def to_batch(imlist):
    if isinstance(imlist, (tuple, list)):
        batch = np.stack(imlist, axis=0)
    elif isinstance(imlist, np.ndarray):
        assert imlist.ndim == 3
        batch = imlist[np.newaxis, :, :, :]
    return batch.astype(np.float32)


class PostProcess:
    def __init__(self, size, mode='yolov8'):
        # [h, w]
        self.size = size
        if mode == 'yolov8':
            self.anchors = 8400
        else:
            raise NotImplementedError(f'mode {mode} does not implement.')
        
    def run(self, im0, predictions, score_threshold, nms_threshold):
        bboxes = predictions[:, :4]
        scores = predictions[:, 4:]
        classes_id = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)
        
        ids = np.where(scores > score_threshold)
        bboxes = bboxes[ids]
        scores = scores[ids]
        classes_id = classes_id[ids]

        bboxes = self.xywh2xyxy(bboxes)
        # bboxes = self.scale_coords(im0, bboxes)
        results = self.multiclass_nms(im0, bboxes, scores, classes_id, nms_threshold)
        return self._format_result(results)
       
    def _format_result(self, result):
        outputs = [[bbox.tolist(), score, cid] for cid, item in result.items() for bbox, score in zip(*item)] 
        return list(zip(*outputs))
    
    def multiclass_nms(self, im0, bboxes, scores, classes_id, nms_threshold):
        unique_classes_id = np.unique(classes_id)
        outputs = {}
        for cid in unique_classes_id:
            coords = bboxes[np.where(classes_id == cid)]
            scores_ = scores[np.where(classes_id == cid)]
            ids = self.nms(coords, scores_, nms_threshold)
            coords = self.scale_coords(im0, coords[ids])
            outputs[cid] = (coords, scores[ids])
        
        return outputs

    def nms(self, coords, scores, nms_threshold):
        orders = np.argsort(scores)[::-1]
        
        keeps = []
        while orders.size:
            idx = orders[0]
            keeps.append(idx)
            
            iou = self.compute_iou(coords[idx], coords[orders[1:]])
            ids = np.where(iou < nms_threshold)[0]
            orders = orders[ids + 1]
        return keeps
    
    def compute_iou(self, bbox, bboxes):
        area_a = np.prod([bbox[2] - bbox[0], bbox[3] - bbox[1]])
        area_b = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        
        xmin = np.maximum(bbox[0], bboxes[:, 0])
        ymin = np.maximum(bbox[1], bboxes[:, 1])
        xmax = np.minimum(bbox[2], bboxes[:, 2])
        ymax = np.minimum(bbox[3], bboxes[:, 3])
        
        inter = np.maximum(0, (ymax- ymin) * (xmax -xmin))
        return inter / (area_a + area_b - inter)
    
    def scale_coords(self, im0, coords):
        h0, w0 = im0.shape[:2]
        r = min(self.size[0] / h0, self.size[1] / w0)
        pad_h, pad_w = (self.size[0] - r * h0) / 2, (self.size[1] - r * w0) / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        
        coords[:, [0, 2]] -= left
        coords[:, [1, 3]] -= top
        coords[:, :4] /= r
        coords[:, 0].clip(0, w0)
        coords[:, 1].clip(0, h0)
        coords[:, 2].clip(0, w0)
        coords[:, 3].clip(0, h0)
        return coords

    def clip_coords(self, boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    
    def xywh2xyxy(self, bboxes):
        # [xc, yc, w, h]
        coords = np.copy(bboxes)
        coords[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        coords[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
        coords[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
        coords[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
        return coords
    

    