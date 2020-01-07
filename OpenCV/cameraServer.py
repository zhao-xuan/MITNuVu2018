import socket
import cv2
import rpistream.camera
import io
import numpy as np
from tempfile import TemporaryFile
import zstandard
import atexit
from rpistream.netutils import *


class Server:
    def __init__(self, **kwargs):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((kwargs.get("bindto", ""), kwargs.get("port", 8080)))
        s.listen(10)
        self.s = s
        self.verbose = kwargs.get("verbose", True)
        atexit.register(self.close)
    def log(self,m):
        if self.verbose:
            print(m)
    def serve(self):
        """Find client"""
        self.log("Searching for client...")
        while True:
            self.conn, self.clientAddr = self.s.accept()
            self.log('Connected with ' + self.clientAddr[0] + ':' + str(self.clientAddr[1]))
            return None

    def startStream(self, getFrame, args=[]):
        """ Creates videostream, calls getFrame to recieve new frames
        Args:
            getFrame: Function executed to generate image frame 
            args: the argumetns passed to the getFrame function
        Returns:
            void
        """
        # send initial frame
        Sfile = io.BytesIO()
        C = zstandard.ZstdCompressor()
        prevFrame = getFrame(*args)
        np.save(Sfile, prevFrame)
        send_msg(self.conn, C.compress(Sfile.getvalue()))
        frameno = 0
        while True:
            Tfile = io.BytesIO()
            # fetch the image
            #print ("Fetching frame...")
            img = getFrame(*args)
            # use numpys built in save function to diff with prevframe
            # because we diff it it will compress more
            np.save(Tfile, img-prevFrame)
            # compress it into even less bytes
            b = C.compress(Tfile.getvalue())
            # reassing prev frame
            prevFrame = img
            # send it
            send_msg(self.conn, b)
            if self.verbose:
                self.log("Sent {}KB (frame {})".format(int(len(b)/1000),frameno))
                frameno+=1
    def close(self):
        """Close all connections"""
        self.s.close()


def retrieveImage(cam, imgResize):
    """Basic function for retrieving camera data, for getFrame"""
    image = cv2.resize(cam.image, (0, 0), fx=imgResize, fy=imgResize)
    return image


if __name__ == "__main__":
    cam = rpistream.camera.Camera(mirror=True)
    resize_cof = 1  # 960p
    server = Server(port=5000)
    server.serve()
    server.startStream(retrieveImage, [cam, resize_cof])