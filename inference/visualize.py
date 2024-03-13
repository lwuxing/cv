from typing import List, Union
import cv2
import numpy as np

ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def gen_colors(num_colors):
    """Generate different colors.

    # Arguments
      num_colors: total number of colors/classes.

    # Output
      bgrs: a list of (B, G, R) tuples which correspond to each of
            the colors/classes.
    """
    import random
    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs


class BBoxVisualize:

    def __init__(self, classes: Union[int, List[str]], show_label=False, show_score=False, margin=2, thickness=1) -> None:
        if isinstance(classes, int):
            classes = list(range(classes))
        
        self.margin = margin  
        self.thickness = thickness
        self.show_label = show_label
        self.show_score = show_score
          
        # self.palette = np.random.uniform(0, 255, (len(classes), 3))
        self.palette = gen_colors(len(classes))
        self.classes_map = {idx: label for idx, label in enumerate(classes)}

    def draw(self, image: np.ndarray, bboxes: List[List[float]], scores: List[float], classes_ids: List[int]) -> np.ndarray:
        """Draw detected bounding boxes on image."""

        img = image.copy()
        for bb, cf, cl in zip(bboxes, scores, classes_ids):
            cl = int(cl)
            bb = [int(b) for b in bb]
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            color = self.palette[cl]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness=self.thickness, lineType=cv2.LINE_AA)
            
            txt = ''
            if self.show_label:
                cls_name = self.classes_map.get(cl, 'CLS{}'.format(cl))
                txt += '{}'.format(cls_name)
            if self.show_score: 
                txt += ' {:.2f}'.format(cf)

            (tw, th), bottom  = cv2.getTextSize(txt, FONT, TEXT_SCALE, TEXT_THICKNESS)
            cv2.rectangle(img, (x_min, y_min), (x_min + tw, y_min + th + bottom), color, -1)
            cv2.putText(img, txt, (int(x_min), int(y_min + th + self.margin)), FONT, TEXT_SCALE, WHITE, 
                        thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
        
        return cv2.addWeighted(img, ALPHA, img, 1 - ALPHA, 0)

        
