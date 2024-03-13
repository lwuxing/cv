from typing import List, Tuple

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from visualize import BBoxVisualize
from ops import Compose, Resize, ToTensor, PostProcess, to_batch


class HostDeviceMemory:
    def __init__(self, host, device):
        self.host = host
        self.device = device
        
    def __del__(self):
        del self.host
        del self.device
    
    
class TRTModel:
    def __init__(self, model_file: str, inp_size=(640, 640)) -> None:
        self.compose = Compose([Resize(letterbox=True), ToTensor()])
        self.inp_size = inp_size
        self.postprocess = PostProcess(size=inp_size)
        self.vis = BBoxVisualize(81)
        self.initialize_model(model_file)
    
    def initialize_model(self, model_file: str) -> None:
        logger = trt.Logger(trt.Logger.INFO)
        with open(model_file, "rb") as f,  trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
    def allocate_buffers(self) -> Tuple[List, List, List]:
        inputs, outputs, bindings = [], [], []
        for binding in self.engine:
            dims = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(dims)
            
            host_memory = cuda.pagelocked_empty_like(np.empty(size, dtype))
            device_memory = cuda.mem_alloc_like(host_memory)
            
            bindings.append(int(device_memory))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMemory(host_memory, device_memory))
            else:
                outputs.append(HostDeviceMemory(host_memory, device_memory))
        return inputs, outputs, bindings

    def preprocess(self, imlist: List[np.ndarray]) -> np.ndarray:
        batch = []
        for im in imlist:
            batch.append(self.compose(im))
        return to_batch(batch)
    
    def __call__(self, 
                 image: np.ndarray, 
                 score_threshold: float=0.25, 
                 nms_threshold: float=0.5, 
                 show: bool=True) -> List:
        # self.context.set_binding_shape(0, (1, 3, self.inp_size[0], self.inp_size[1]))
        inputs, outputs, bindings = self.allocate_buffers()
        
        im = self.preprocess([image])
        np.copyto(inputs[0].host, im.ravel())
        cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, self.stream)
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, self.stream)
        self.stream.synchronize()
        
        output = outputs[0].host.reshape(-1, 84, 8400)
        output = np.squeeze(output, axis=0).transpose(-1, -2)
        result = self.postprocess.run(image, output, score_threshold, nms_threshold)
        
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
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    image = cv2.imread("000000000.jpg")
    model = TRTModel("yolov8m.plan")
    model(image)

