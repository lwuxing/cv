import time
import traceback
from queue import Queue
from threading import Thread, Lock

import cv2
import numpy as np


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
        print(f"The total number of frames of the video stream is {self.frames[0]}")
        self.imgs[0] = frame

    def _read(self, transform=False):
        im0 = self.imgs.copy()

        return im0, None

    def read(self, i=0, transform=True):
        n, f = 0, self.frames[i]
        max_retries = 10  # 设置最大重试次数
        retries = 0
        
        while n < f:
            try:
                if not self.cap.isOpened():
                    print(f'WARNING: Video stream {i + 1} is not opened, retrying...')
                    self.cap.open(self.sources[i])
                    assert self.cap.isOpened(), f'Failed to open {self.sources[i]} after retrying'

                n += 1
                self.cap.grab()
                if n % self.vid_stride == 0 or n == f:
                    success, im = self.cap.retrieve()
                    if success:
                        self.imgs[i] = im
                    else:
                        print(f'WARNING: Video stream {i + 1} unresponsive, retrying...')
                        self.cap.open(self.sources[i])  # re-open stream if signal was lost
                        assert self.cap.isOpened(), f'Failed to open {self.sources[i]} after lost'
                        _, frame = self.cap.read()
                        self.imgs[i] = frame

                    yield self._read(transform)
            except Exception as e:
                print(f'Error occurred: {traceback.format_exc()}')
                retries += 1
                if retries < max_retries:
                    print(f'Wait for 3 sec, Retrying ({retries}/{max_retries})...')
                    time.sleep(3)
                    self.cap.open(self.sources[i])
                else:
                    print(f'Max retries reached. Unable to recover.')
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
                    print(f'[!] WARNING: Video stream {i + 1} unresponsive, please check your connection.')
                    with self.locks[i]:
                        shape = self.hw[i] + (3, )
                        self.imgs_queues[i].put(np.zeros_like(np.empty(shape)))  # Put a black frame if retrieval fails
                    cap.open(self.sources[i])  # re-open stream if signal was lost

        print(f'[!] WARNING: Video stream close, due to n is {n} and f is {f}')
        self.imgs_queues[i].put(None)

    def read(self, transform=False):
        if True:
            im0 = [self.get_nonzero_frame(i) for i in range(self.num_sources)]

            if any(im is None for im in im0):
                return None, None
        else:
            im0 = [self.imgs_queues[i].get() for i in range(self.num_sources)]

        return im0, None

    def check_threads_active(self):
        # Check if all threads are still alive and successfully read a frame
        # return all(thread.is_alive() and not np.all(img == 0) for thread, img in zip(self.threads, self.imgs))
        print("Thread Status:")
        for i, thread in enumerate(self.threads):
            print(f"Thread {i + 1}: Alive={thread.is_alive()}, Frame Status: {not np.all(self.frames[i] == 0)}")
        print("All threads have finished or are not streaming.")


    def get_nonzero_frame(self, i):
        frame = self.imgs_queues[i].get()
        if frame is None:
            print(f'[!] WARNING: Video stream {i} received None value, ending read.')
            return None
        if not np.all(frame == 0):
            return frame

        print(f'[!] WARNING: Video stream {i + 1} has no valid frame after attempts.')
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
                    print(f'WARNING: Video stream {i + 1} unresponsive, please check your connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(self.sources[i])  # re-open stream if signal was lost

    def check_threads_active(self):
        # Check if all threads are still alive and successfully read a frame
        # return all(thread.is_alive() and not np.all(img == 0) for thread, img in zip(self.threads, self.imgs))
        print("Thread Status:")
        for i, thread in enumerate(self.threads):
            print(f"Thread {i + 1}: Alive={thread.is_alive()}, Frame Status: {not np.all(self.imgs[i] == 0)}")
        print("All threads have finished or are not streaming.")

    
    def _read(self, transform=False):
        self.check_threads_active()
        im0 = self.imgs.copy()

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
        print(frames[1].shape)
        cv2.imshow('test', frames[1])

        # Exit condition (for example, press 'q' key to quit)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
