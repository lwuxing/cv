
"""
华为
"""
import json
import logging
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, MxProtobufIn, StringVector
from typing import List
import cv2

import numpy as np

from visualize import BBoxVisualize
from ops import Compose, Resize, ToTensor, PostProcess, to_batch

class AscendApi:
    """
    Manage pieline stream
    """
    def __init__(self, pipeline_cfg, stream_name, infer_timeout=100000):
        """
        Parameter initialization
        """
        self.pipeline_cfg = pipeline_cfg
        self._stream_api = None
        self._data_input = None
        self._device_id = None
        self.stream_name = stream_name
        self.infer_timeout = infer_timeout

    def init(self):
        """
        Stream initialization
        """
        with open(self.pipeline_cfg, 'r') as fp:
            self._device_id = int(json.loads(fp.read())[self.stream_name]["stream_config"]["deviceId"])

            print(f"The device id: {self._device_id}.")

        # create api
        self._stream_api = StreamManagerApi()

        # init stream mgr
        ret = self._stream_api.InitManager()
        if ret != 0:
            print(f"Failed to init stream manager, ret={ret}.")
            return False

        # create streams
        with open(self.pipeline_cfg, 'rb') as fp:
            pipe_line = fp.read()

        ret = self._stream_api.CreateMultipleStreams(pipe_line)
        if ret != 0:
            print(f"Failed to create stream, ret={ret}.")
            return False

        self._data_input = MxDataInput()
        return True

    def __del__(self):
        if not self._stream_api:
            return

        self._stream_api.DestroyAllStreams()

    def send_data_input(self, stream_name, plugin_id, input_data):
        data_input = MxDataInput()
        data_input.data = input_data
        unique_id = self._stream_api.SendData(stream_name, plugin_id,
                                              data_input)
        if unique_id < 0:
            logging.error("Fail to send data to stream.")
            return False
        return True

    def get_protobuf(self, stream_name, plugin_id, keys):
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)

        infer_result = self._stream_api.GetProtobuf(stream_name, plugin_id, keyVec)

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)

        return result

    def _send_protobuf(self, stream_name, plugin_id, element_name, buf_type,
                       pkg_list):
        """
        Input image data
        """
        protobuf = MxProtobufIn()
        protobuf.key = element_name.encode("utf-8")
        protobuf.type = buf_type
        protobuf.protobuf = pkg_list.SerializeToString()
        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf)
        err_code = self._stream_api.SendProtobuf(stream_name, plugin_id,
                                                 protobuf_vec)
        if err_code != 0:
            logging.error(
                "Failed to send data to stream, stream_name(%s), plugin_id(%s), element_name(%s), "
                "buf_type(%s), err_code(%s).", stream_name, plugin_id,
                element_name, buf_type, err_code)
            return False
        return True

    def send_img_input(self, stream_name, plugin_id, element_name, input_data,
                       img_size):
        """
        input image data after preprocess
        """
        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionInfo.format = 1
        vision_vec.visionInfo.width = img_size[1]
        vision_vec.visionInfo.height = img_size[0]
        vision_vec.visionInfo.widthAligned = img_size[1]
        vision_vec.visionInfo.heightAligned = img_size[0]
        vision_vec.visionData.memType = 0
        vision_vec.visionData.dataStr = input_data
        vision_vec.visionData.dataSize = len(input_data)

        buf_type = b"MxTools.MxpiVisionList"
        return self._send_protobuf(stream_name, plugin_id, element_name,
                                   buf_type, vision_list)

    def send_tensor_input(self, stream_name, plugin_id, element_name,
                          input_data, input_shape, data_type):
        """
        get image tensor
        """
        tensor_list = MxpiDataType.MxpiTensorPackageList()
        tensor_pkg = tensor_list.tensorPackageVec.add()
        # init tensor vector
        tensor_vec = tensor_pkg.tensorVec.add()
        tensor_vec.deviceId = self._device_id
        tensor_vec.memType = 0
        tensor_vec.tensorShape.extend(input_shape)
        tensor_vec.tensorDataType = data_type
        tensor_vec.dataStr = input_data
        tensor_vec.tensorDataSize = len(input_data)

        buf_type = b"MxTools.MxpiTensorPackageList"
        return self._send_protobuf(stream_name, plugin_id, element_name,
                                   buf_type, tensor_list)



class OMModel:

    def __init__(self, pipeline_file: str) -> None:
        self.compose = Compose([Resize(letterbox=True), 
                                ToTensor(),
                                # Normalize()
                            ])
        
        self.postprocess = PostProcess(size=(640, 640))
        self.vis = BBoxVisualize(81)
        self.initialize_model(pipeline_file)

        self.tensor_dtype_map = {'float32': 0, 'float16':1, 'int8': 2}
        self.img_data_plugin_id = 0
        
    def initialize_model(self, pipeline_file: str, stream_name: str) -> None:
        self.api = AscendApi(pipeline_file, stream_name)
        if not self.api.init():
            raise RuntimeError("[!] sdk api failed to initialize")

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
        im = self.preprocess([image])
        self.api.send_tensor_input(self.api.stream_name.encode("utf-8"), 
                                    self.img_data_plugin_id, 
                                    "appsrc0", 
                                    im.tobytes(), 
                                    im.shape, 
                                    self.tensor_dtype_map['float32'])
        keys = [b"mxpi_tensorinfer0"]
        api_result = self.api.get_protobuf(self.api.stream_name.encode("utf-8"), 
                                           self.img_data_plugin_id, 
                                           keys)
        
        # YOLOV8 output is [N, clsNum+4, 8400], YOLOV5 is [N，25200，clsNum+5]
        _dims = len(self.config['ObjectClass']) + 4
        outputs = np.frombuffer(
            api_result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='float32')
        predictions = outputs.reshape(-1, 8400).T
        
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

    model = OMModel("om.pipeline")
    model(image)