import cv2
import asyncio
import queue
import threading
from av import VideoFrame
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError

async def visualize_cv2(track, name=None):
    frame = await track.recv()
    img = frame.to_ndarray(format="bgr24")
    name = name if name else "Video Image Track"
    cv2.imshow(name, img)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
    return frame


async def visualize_cv2_task(track, name=None):
    while True:
        try:
            await visualize_cv2(track, name=name)
        except MediaStreamError:
            return

class IdentityStreamTrack(VideoStreamTrack):
    """MediaStreamTrack to return the original frame as is
    """
    kind = 'video'
    
    def __init__(self, track):
        super().__init__()
        self.track = track
    
    async def recv(self):
        frame = await self.track.recv()
        return frame

class ModelStreamTrack(VideoStreamTrack):
    """MediaStreamTrack to return the original frame passed through some diffusion models
    """
    kind = 'video'
    
    def __init__(self, track, model):
        super().__init__()
        self.track = track
        self.model = model
        self.frame_queue = queue.Queue()
        self.output_lock = threading.Lock()
        self.output = None
        self.t = threading.Thread(target=self._processor)
        self.t.daemon = True
        self.t.start()
    
    def _processor(self):
        while True:
            frame = self.frame_queue.get()
            img = frame.to_ndarray(format="bgr24")
            if self.model:
                img = self.model.forward(img)
            
            with self.output_lock:
                self.output = img


    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = await self.track.recv()
        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

        with self.output_lock:
            if not self.output is None:
                output = self.output
            else:
                output = frame.to_ndarray(format="bgr24")
        return_frame = VideoFrame.from_ndarray(output)
        return_frame.pts = pts
        return_frame.time_base = time_base
        return return_frame

class VisualizeStreamTrack(VideoStreamTrack):
    """MediaStreamTrack to show the frames on a cv2 window
    """
    kind = 'video'

    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        return visualize_cv2(self.track)

class MediaVisualizer:
    """Show the frames on a cv2 window
    """
    def __init__(self):
        self.__tracks = {}

    def addTrack(self, track):
        if track not in self.__tracks:
            self.__tracks[track] = None

    async def start(self):
        for i, (track, task) in enumerate(self.__tracks.items()):
            if task is None:
                self.__tracks[track] = asyncio.ensure_future(visualize_cv2_task(track, "Video " + str(i)))

    async def stop(self):
        for task in self.__tracks.values():
            if task is not None:
                task.cancel()
        self.__tracks = {}
