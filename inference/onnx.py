from typing import List
import cv2
import onnxruntime
import numpy as np

from visualize import BBoxVisualize
from ops import Compose, Resize, ToTensor, PostProcess, to_batch

    
class ONNXModel:
    
    def __init__(self, model_file: str) -> None:
        self.compose = Compose([Resize(letterbox=True), 
                                ToTensor(),
                                # Normalize()
                            ])
        
        self.postprocess = PostProcess(size=(640, 640))
        self.vis = BBoxVisualize(81)
        self.initialize_model(model_file)
        
    def preprocess(self, imlist: List[np.ndarray]) -> np.ndarray:
        batch = []
        for im in imlist:
            batch.append(self.compose(im))
        return to_batch(batch)
    
    def initialize_model(self, model_file: str) -> None:
        provider = onnxruntime.get_available_providers()
        self.session = onnxruntime.InferenceSession(model_file, providers=provider)

        oup = self.session.get_outputs()
        self.oup = [oup[i].name for i in range(len(oup))]

        inp = self.session.get_inputs()
        self.inp = [inp[i].name for i in range(len(inp))]

    def __call__(self, 
                 image: np.ndarray, 
                 score_threshold: float=0.25, 
                 nms_threshold: float=0.5, 
                 show: bool=True) -> List:
        im = self.preprocess([image])
        output = self.session.run(output_names=self.oup, input_feed={self.inp[0]: im})
        predictions = np.transpose(np.squeeze(output[0]))
        result = self.postprocess.run(image, predictions, score_threshold, nms_threshold)
        
        if show:
            self.to_show(image, result[0], result[1], result[2])
        
        return result
        
    def to_show(self, 
                image: np.ndarray, 
                bboxes: List[List[float]], 
                scores: List[float], 
                classes_id: List[int]) -> None:
        img = self.vis.draw(image, bboxes, scores, classes_id)
        cv2.imwrite('show.jpg', img)
        
    
if __name__ == "__main__":
    image = cv2.imread("000000000.jpg")

    model = ONNXModel("yolov8m.onnx")
    model(image)
