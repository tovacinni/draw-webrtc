import sys
import argparse
import asyncio
import websockets

import cv2
import numpy as np

from mss import mss
import pywinctl as pwc
import time
import logging

from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.signaling import BYE, add_signaling_arguments
from aiortc.contrib.media import MediaRelay

from av import VideoFrame

from drawrtc.signaling import WebsocketSignaling
from drawrtc.mediastream import MediaVisualizer
from drawrtc.config import rtc_configuration

handlers = [logging.StreamHandler(sys.stdout)]
logging.basicConfig(level=logging.INFO,
                    #format='%(asctime)s|%(levelname)8s| %(message)s',
                    handlers=handlers)

class ScreenRecordingStreamTrack(VideoStreamTrack):
    kind = "video"
    def __init__(self):
        super().__init__()
        self.sct = mss()

        # Currently hacked to work for a movie recording stream from iPads
        print("Finding QuickTime stream... (this will take a few seconds for some unknown reasons)")
        try:
            idx = pwc.getAllTitles().index("Movie Recording")
        except:
            print("ERROR: Movie Recording not found. Exiting.")
            sys.exit()

        active_window = pwc.getAllWindows()[idx]
        pass
        print("Quick time stream found!")

        self.bbox = {
            "top": active_window.top,
            "left": active_window.left,
            "width": active_window.width,
            "height": active_window.height
        }

        # Do a square crop
        new_width = int(self.bbox["width"] * 0.9)
        self.bbox["left"] += (self.bbox["width"] - new_width) // 2
        self.bbox["width"] = new_width
        top_offset = (self.bbox["height"] - self.bbox["width"]) // 2
        self.bbox["top"] += top_offset
        self.bbox["height"] = self.bbox["width"]

    def screenshot(self):
        """Takes a screenshot using the bounding box.

        Returns:
            (np.ndarray): a [h, w, 4] array for the current frame
        """
        screenshot = self.sct.grab(self.bbox)
        arr = np.array(screenshot)
        return arr

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(self.screenshot()[..., :3])
        frame.pts = pts
        frame.time_base = time_base
        return frame

async def run(pc, signaling):
    # connect signaling
    await signaling.connect()
    visualizer = MediaVisualizer()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            visualizer.addTrack(track)
            pass

    # send offer
    pc.addTrack(ScreenRecordingStreamTrack())
    await pc.setLocalDescription(await pc.createOffer())
    await signaling.send(pc.localDescription)

    while True:
        await visualizer.start()

        obj = await signaling.receive()

        if isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)
        elif isinstance(obj, RTCIceCandidate):
            await pc.addIceCandidate(obj)
        elif obj is BYE:
            print("Exiting")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drawing tool with WebRTC")
    parser.add_argument("--show-stream", action="store_true", 
                        help="Shows the stream being sent in an OpenCV window")
    add_signaling_arguments(parser)
    args = parser.parse_args()

    if args.show_stream:
        stream = ScreenRecordingStreamTrack()
        count = 0
        while True:
            count += 1

            frame = stream.screenshot()
            cv2.imshow("Studio Output", frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()

    else:
        signaling = WebsocketSignaling(host=args.signaling_host, port=args.signaling_port)
        pc = RTCPeerConnection(configuration=rtc_configuration)
        loop = asyncio.get_event_loop()

        try:
            loop.run_until_complete(
                run(pc=pc, signaling=signaling)
            )
        except KeyboardInterrupt:
            pass
        finally:
            loop.run_until_complete(signaling.close())
            loop.run_until_complete(pc.close())
            pass
