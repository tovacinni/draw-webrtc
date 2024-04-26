import os
import sys
import asyncio
import argparse
import cv2
import numpy as np
import time
import contextlib

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaRelay
from aiortc.contrib.signaling import BYE
import uuid
import websockets
import logging
from drawrtc.signaling import WebsocketSignaling, object_to_string, object_from_string
from drawrtc.config import rtc_configuration
from drawrtc.mediastream import ModelStreamTrack

handlers = [logging.StreamHandler(sys.stdout)]
logging.basicConfig(level=logging.INFO,
                    #format='%(asctime)s|%(levelname)8s| %(message)s',
                    handlers=handlers)

logger = logging.getLogger("pc")
relay = MediaRelay()
pcs = set()

if sys.platform == 'linux':
    from drawrtc.models import ImageToImage
    model = ImageToImage()
else:
    model = None


async def signaling_handler(websocket):
    pc = RTCPeerConnection(configuration=rtc_configuration)
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)
    
    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        if track.kind == "video":
            pc.addTrack(ModelStreamTrack(track, model))

    while True:
        
        # Read the object that was sent
        try:
            obj = await websocket.recv()
            obj = object_from_string(obj)
            if obj == None:
                log_info("Remote host says good bye!")
        except asyncio.IncompleteReadError:
            logging.error("Read error")
            await pc.close()
            pcs.discard(pc)
            return

        if isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)
            if obj.type == "offer":
                log_info("Offer received")
                # send answer
                # pc.addTrack()
                await pc.setLocalDescription(await pc.createAnswer())
                await websocket.send(object_to_string(pc.localDescription) + '\n')
        elif isinstance(obj, RTCIceCandidate):
            log_info("ICE candidate received")
            await pc.addIceCandidate(obj)
        elif obj is BYE:
            print("Exiting")
            break


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def main():
    async with websockets.serve(signaling_handler, "localhost", 8080):
        await asyncio.Future()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drawing tool with WebRTC")
    parser.add_argument("--disable-model", action="store_true", 
                        help="Disables the diffusion forward pass")
    args = parser.parse_args()
    if args.disable_model:
        model = None
    asyncio.run(main())
