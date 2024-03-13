
import io
import os
import cv2
import math
import time
import yaml
import requests
import logging
import traceback
import numpy as np

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from logging import handlers
from PIL import Image
from queue import Queue
from threading import Thread, Lock
from datetime import datetime, timedelta, timezone

VID_STRIDE = 3
__logger = []
log = logging.getLogger('linewell')


class TrackingManager:
    def __init__(self):
        self.tracking_history = {}
        self.stopped_ids_cache = set()  # 用于存储已经判断为静止的ID

    @staticmethod
    def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

        # Get the coordinates of bounding boxes
        if xywh:
            # transform from xywh to xyxy
            x1, y1, w1, h1 = np.split(box1, 4, axis=-1)
            x2, y2, w2, h2 = np.split(box2, 4, axis=-1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:
            # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = np.split(box1, 4, axis=-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = np.split(box2, 4, axis=-1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        inter = np.maximum(0, (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1))) * \
                np.maximum(0, (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)))

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + eps

        # IoU
        iou = inter / union
        if CIoU or DIoU or GIoU:
            cw = np.maximum(0, np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1))  # convex (smallest enclosing box) width
            ch = np.maximum(0, np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1))  # convex height
            if CIoU or DIoU:
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
                if CIoU:
                    v = (4 / np.pi ** 2) * (np.arctan(w2 / h2) - np.arctan(w1 / h1)) ** 2
                    alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                return iou - rho2 / c2  # DIoU
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        return iou  # IoU

    def update_tracking_history(self, tracking_id, tracking_data):
        current_time = datetime.now()

        if tracking_id not in self.tracking_history:
            self.tracking_history[tracking_id] = {'data': tracking_data, 'timestamp': current_time, 'iou_flag': False, 
                                                  'consecutive_true_count': 0, 'need_save_frame': False, 'first_iou_true_timestamp': None, 
                                                  'iou_with_previous': None, 'iou_with_first': None, 'first_bbox': tracking_data['bbox']}
        else:
            previous_bbox = self.tracking_history[tracking_id]['data']['bbox']
            current_bbox = tracking_data['bbox']
            iou_with_previous = round(self.bbox_iou(previous_bbox, current_bbox, xywh=False, DIoU=True).tolist()[0], 3)
            iou_with_first = round(self.bbox_iou(self.tracking_history[tracking_id]['first_bbox'], current_bbox, xywh=False, DIoU=True).tolist()[0], 3)

            self.tracking_history[tracking_id]['iou_flag'] = (iou_with_previous > 0.9) and (iou_with_first > 0.9)

            if self.tracking_history[tracking_id]['iou_flag']:
                if self.tracking_history[tracking_id]['consecutive_true_count'] == 0:
                    self.tracking_history[tracking_id]['first_iou_true_timestamp'] = current_time
                self.tracking_history[tracking_id]['consecutive_true_count'] += 1

                if self.tracking_history[tracking_id]['consecutive_true_count'] >= 10:
                    self.tracking_history[tracking_id]['need_save_frame'] = True
            else:
                self.tracking_history[tracking_id]['consecutive_true_count'] = 0
                self.tracking_history[tracking_id]['need_save_frame'] = False

            # 更新数据、时间戳和IoU状态
            self.tracking_history[tracking_id]['data'] = tracking_data
            self.tracking_history[tracking_id]['timestamp'] = current_time
            self.tracking_history[tracking_id]['iou_with_previous'] = iou_with_previous
            self.tracking_history[tracking_id]['iou_with_first'] = iou_with_first

    def is_object_stopped(self):
        current_time = datetime.now()
        # threshold_time = timedelta(seconds=10*VID_STRIDE)
        threshold_time = timedelta(seconds=0.3*VID_STRIDE)

        for tracking_id in self.tracking_history:

            if tracking_id in self.stopped_ids_cache or \
               (not self.tracking_history[tracking_id]['need_save_frame']) or \
                (self.tracking_history[tracking_id]['need_save_frame'] and
                 (current_time - self.tracking_history[tracking_id]['first_iou_true_timestamp']) < threshold_time):
                continue

            if self.tracking_history[tracking_id]['iou_flag']:
                yield {tracking_id: self.tracking_history[tracking_id]}
                self.stopped_ids_cache.add(tracking_id)


    def remove_inactive_entries(self):
        current_time = datetime.now()
        # expiration_time = timedelta(seconds=30*VID_STRIDE)
        expiration_time = timedelta(seconds=0.5*VID_STRIDE)

        inactive_ids = [tracking_id for tracking_id, info in self.tracking_history.items()
                        if (current_time - info['timestamp']) > expiration_time]

        for tracking_id in inactive_ids:
            del self.tracking_history[tracking_id]

        needed_to_save = []
        for tracking_data in self.is_object_stopped():
            needed_to_save.append(tracking_data)
        
        return needed_to_save


class FrameUpLoader:
    def __init__(self, tracking_manager: TrackingManager, upload_url, event_url, tag='游商小贩'):
        self.tracking_manager = tracking_manager
        self.upload_url = upload_url
        self.event_url = event_url
        self.tag = tag

    def process_frame(self, im0):
        needed = self.tracking_manager.remove_inactive_entries()
        # log.info(f"Entries to upload: {needed}")
        if len(needed) != 0:
            log.info(f"Entries to upload: {needed}")
            self.simple_upload(im0)

            return time.time()

        return None

    def simple_upload(self, im0):
        try:
            current_time = datetime.now(timezone.utc)
            timestamp = current_time.strftime("%Y%m%d%H%M%S")
            dt_object = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
            iso_timestamp = dt_object.isoformat()

            with io.BytesIO() as imfile:
                img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img.save(imfile, format='JPEG')
                imfile.seek(0)
                img_bytes = imfile.read()

            headers = {
                'Accept': '*/*',
                'Body-Check': 'bc4e3f3b9939bb9549cd84a70515e8c3'
            }

            files = {'file': ('1.jpg', img_bytes, 'image/jpg')}
            response = requests.post(self.upload_url, files=files, headers=headers)
            datas = response.json()

            if datas['code'] == 200:
                img_url = datas["data"]
                log.info(img_url)

            data = {
                "algorithmType": "",
                "bbox": "[0,0,0,0]",
                "deviceId": 2,
                "eventName": f"{self.tag}{timestamp}",
                "eventTime": iso_timestamp,
                "eventType": f"{self.tag}",
                "imageUrl": img_url
            }

            response = requests.post(self.event_url, json=data, headers=headers)
            log.info("===================== upload success =====================")
        except Exception as _ex:
            log.info("===================== upload fail =====================")
            log.error(traceback.format_exc())

class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, Mapping):
                self[key] = EasyDict(value)

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, Mapping):
                return EasyDict(value)
            else:
                return value
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class YamlParser(EasyDict):
    # yaml parser based on custom EasyDict.

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        special = ['project', 'reid_weights', 'tracking_config', 'yolo_weights']
        
        if config_file is not None:
            assert os.path.isfile(config_file), f"paddle ocr config file '{config_file}' doesn't find"
            with open(config_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            
            for k, v in yaml_.items():
                if k in special:
                    yaml_[k] = Path(v)
            cfg_dict.update(yaml_)

        # if os.environ.get('source', None):
        #     cfg_dict['source'] = os.environ.get('source')
        if os.environ.get('threshold', None):
            threshold = float(os.environ.get('threshold'))
            assert threshold > 0.0 and threshold < 1.0
            cfg_dict['conf_thres'] = float(os.environ.get('threshold'))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


@lru_cache(maxsize=128) 
def config_from_file_or_dict(cfg_file=None, cfg_dict=None):
    assert cfg_file or cfg_dict
    return YamlParser(config_file=cfg_file, cfg_dict=cfg_dict)


_cfg = config_from_file_or_dict(cfg_file="data.yml")


def get_logger(name="linewell", output=None, debug=True):
    logger = logging.getLogger(name)

    if name in __logger:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        '[%(asctime)s %(levelname)s %(filename)s line:%(lineno)d] - %(message)s')

    # stdout logging: debug only
    if debug:
        import sys
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if output is not None:
        now = datetime.now()
        basename = now.strftime("%Y-%m-%d.log")
        filename = os.path.join(output, basename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 修改这里的 TimedRotatingFileHandler 配置
        fh = handlers.TimedRotatingFileHandler(
            filename, when="MIDNIGHT", backupCount=20  # 设置为备份6个文件，保留6天的日志
        )

        fh.setFormatter(formatter)
        logger.addHandler(fh)

    __logger.append(name)

    return logger


def check_imgsz(imgsz, stride=32, min_dim=1, floor=0):
    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]

    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        log.info(f'WARNING: --img-size {imgsz} must be multiple of max stride {stride}, updating to {sz}')

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:
    # Annotator for detect inference annotations
    def __init__(self, im, line_width=None):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with cv2
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose"""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels"""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class SingleRTSPReader:
    def __init__(self, source, imgsz=640, stride=32, vid_stride=1):
        self.stride = stride
        self.sources = source
        self.imgsz = imgsz
        self.vid_stride = vid_stride
        self.num_sources = 1
        self.frames = [None]
        self.imgs = [None]
        self.cap = cv2.VideoCapture(source[0])
        assert self.cap.isOpened(), f'Failed to open {source[0]}'
        _, frame = self.cap.read()
        self.frames[0] = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
        log.info(f"The total number of frames of the video stream is {self.frames[0]}")
        self.imgs[0] = frame

    def _read(self, transform=False):
        im0 = self.imgs.copy()
        if transform:
            auto = True
            im = np.stack([LetterBox(self.imgsz, auto, stride=self.stride)(image=x) for x in im0])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous
            return im0, im

        return im0, None

    def read(self, i=0, transform=True):
        n, f = 0, self.frames[i]
        max_retries = 10  # 设置最大重试次数
        retries = 0
        
        while n < f:
            try:
                if not self.cap.isOpened():
                    log.info(f'WARNING: Video stream {i + 1} is not opened, retrying...')
                    self.cap.open(self.sources[i])
                    assert self.cap.isOpened(), f'Failed to open {self.sources[i]} after retrying'

                n += 1
                self.cap.grab()
                if n % self.vid_stride == 0 or n == f:
                    success, im = self.cap.retrieve()
                    if success:
                        self.imgs[i] = im
                    else:
                        log.info(f'WARNING: Video stream {i + 1} unresponsive, retrying...')
                        self.cap.open(self.sources[i])  # re-open stream if signal was lost
                        assert self.cap.isOpened(), f'Failed to open {self.sources[i]} after lost'
                        _, frame = self.cap.read()
                        self.imgs[i] = frame

                    yield self._read(transform)
            except Exception as e:
                log.info(f'Error occurred: {traceback.format_exc()}')
                retries += 1
                if retries < max_retries:
                    log.info(f'Wait for 3 sec, Retrying ({retries}/{max_retries})...')
                    time.sleep(3)
                    self.cap.open(self.sources[i])
                else:
                    log.info(f'Max retries reached. Unable to recover.')
                    break


class MultiRTSPReaderWithQueue:
    def __init__(self, sources: list, imgsz=640, stride=32, vid_stride=1):
        self.stride = stride
        self.sources = sources
        self.imgsz = imgsz
        self.vid_stride = vid_stride
        self.num_sources = len(sources)
        self.frames = [None] * self.num_sources
        self.hw = [None] * self.num_sources
        self.imgs_queues = {i: Queue() for i in range(self.num_sources)}
        self.threads = [None] * self.num_sources
        self.locks = [Lock() for _ in range(self.num_sources)]

        for i, source in enumerate(sources):
            cap = cv2.VideoCapture(source)
            assert cap.isOpened(), f'Failed to open {source}'
            _, frame = cap.read()
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            self.hw[i] = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.imgs_queues[i].put(frame)  # Put the initial frame into the queue
            self.threads[i] = Thread(target=self.update, args=(i, cap), daemon=True)
            self.threads[i].start()

    def update(self, i, cap):
        n, f = 0, self.frames[i]
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    with self.locks[i]:
                        self.imgs_queues[i].put(im)  # Put the new frame into the queue
                else:
                    log.info(f'[!] WARNING: Video stream {i + 1} unresponsive, please check your connection.')
                    with self.locks[i]:
                        shape = self.hw[i] + (3, )
                        self.imgs_queues[i].put(np.zeros_like(np.empty(shape)))  # Put a black frame if retrieval fails
                    cap.open(self.sources[i])  # re-open stream if signal was lost

        log.info(f'[!] WARNING: Video stream close, due to n is {n} and f is {f}')
        self.imgs_queues[i].put(None)

    def read(self, transform=False):
        if True:
            im0 = [self.get_nonzero_frame(i) for i in range(self.num_sources)]

            if any(im is None for im in im0):
                return None, None
        else:
            im0 = [self.imgs_queues[i].get() for i in range(self.num_sources)]
        if transform:
            auto = True
            im = np.stack([LetterBox(self.imgsz, auto, stride=self.stride)(image=x) for x in im0])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous
            return im0, im

        return im0, None

    def check_threads_active(self):
        # Check if all threads are still alive and successfully read a frame
        # return all(thread.is_alive() and not np.all(img == 0) for thread, img in zip(self.threads, self.imgs))
        log.info("Thread Status:")
        for i, thread in enumerate(self.threads):
            log.info(f"Thread {i + 1}: Alive={thread.is_alive()}, Frame Status: {not np.all(self.frames[i] == 0)}")
        log.info("All threads have finished or are not streaming.")


    def get_nonzero_frame(self, i):
        frame = self.imgs_queues[i].get()
        if frame is None:
            log.info(f'[!] WARNING: Video stream {i} received None value, ending read.')
            return None
        if not np.all(frame == 0):
            return frame

        log.info(f'[!] WARNING: Video stream {i + 1} has no valid frame after attempts.')
        return None


class MultiRTSPReader:
    def __init__(self, sources: list, imgsz=640, stride=32, vid_stride=1):
        self.stride = stride
        self.sources = sources
        self.imgsz = imgsz
        self.vid_stride = vid_stride
        self.num_sources = len(sources)
        self.frames = [None] * self.num_sources
        self.imgs = [None] * self.num_sources
        self.threads = [None] * self.num_sources

        for i, source in enumerate(sources):
            cap = cv2.VideoCapture(source)
            assert cap.isOpened(), f'Failed to open {source}'
            _, frame = cap.read()
            self.frames[i] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.imgs[i] = frame
            self.threads[i] = Thread(target=self.update, args=(i, cap), daemon=True)
            self.threads[i].start()

    def update(self, i, cap):
        n, f = 0, self.frames[i]
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    log.info(f'WARNING: Video stream {i + 1} unresponsive, please check your connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(self.sources[i])  # re-open stream if signal was lost

    def check_threads_active(self):
        # Check if all threads are still alive and successfully read a frame
        # return all(thread.is_alive() and not np.all(img == 0) for thread, img in zip(self.threads, self.imgs))
        log.info("Thread Status:")
        for i, thread in enumerate(self.threads):
            log.info(f"Thread {i + 1}: Alive={thread.is_alive()}, Frame Status: {not np.all(self.imgs[i] == 0)}")
        log.info("All threads have finished or are not streaming.")

    
    def _read(self, transform=False):
        self.check_threads_active()
        im0 = self.imgs.copy()
        if transform:
            auto = True
            im = np.stack([LetterBox(self.imgsz, auto, stride=self.stride)(image=x) for x in im0])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous
            return im0, im

        return im0, None
    
    def read(self, transform=False):
        while True:
            yield self._read(transform)


if __name__ == "__main__":
    # Example usage
    sources = [
        'rtsp://admin:HikTPALJY@10.231.6.55:554/Streaming/Channels/1',
        'rtsp://admin:HikTPALJY@10.231.6.55:554/Streaming/Channels/1'
    ]

    multi_reader = MultiRTSPReader(sources)

    while True:
        frames = multi_reader.read()

        # Check if threads are still active and streaming
        multi_reader.check_threads_active()

        # Process frames as needed
        log.info(frames[1].shape)
        cv2.imshow('test', frames[1])

        # Exit condition (for example, press 'q' key to quit)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
