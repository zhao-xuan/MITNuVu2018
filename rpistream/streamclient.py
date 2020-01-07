import socket
import numpy as np
import io
import cv2
import zstandard # zstd might work on other computers but only zstandard will work with mine
import atexit
from rpistream.netutils import *
import os.path
import time

class Client:
    def __init__(self, **kwargs):
        self.verbose = kwargs.get("verbose", True)
        # output file seems to be corrupted: likely due to output file stream not being closed correctly
        self.Write = kwargs.get("WriteFile", False)
        self.writepath = kwargs.get("path", "")
        self.FileFPS = kwargs.get("fileoutFps", 10)
        self.FileName = kwargs.get("fileName", 'outpy')
        self.iRes = kwargs.get("imageResolution", (1280, 960))
        self.viewScale = kwargs.get("viewscale", 1.0)
        
        # saves values to create socket
        self.log("Initializing socket...")
        self.ip = kwargs.get("serverIp", "localhost")
        self.port = kwargs.get("port", 8080)
        
        self.viewScale = kwargs.get("viewScale", 1) # the scaling factor for the display


    def connect(self):
        # creates socket
        self.s = socket.socket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        #connects via socket to server at provided ip over the provided port
        self.log("Connecting...")
        self.s.connect((self.ip, self.port)) #connect over port
        # create video codec
        fourcc = None
        try:
            fourcc = cv2.cv.CV_FOURCC(*'MJPG') # OpenCV 2 function
        except:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # OpenCV 3 function
        # write to a new file
        if self.Write:
            counter = 0
            while True:
                newfile = self.writepath+self.FileName+str(counter)+'.avi'
                if os.path.exists(newfile):
                    counter = counter + 1
                else:
                    self.out = cv2.VideoWriter(newfile, fourcc, self.FileFPS, self.iRes)
                    break
                
        # create compressor
        self.D = zstandard.ZstdDecompressor() #instanciate a decompressor which we can use to decompress our frames
        self.log("Ready")
        
    def log(self, m):
        if self.verbose:
            print(m) #printout if server is in verbose mode

    def recv(self, size=1024):
        """Recieves a single frame
        Args:
            size: how big a frame should be
                default: 1024
        returns:
            single data frame
        """
        data = bytearray()
        while 1:
            buffer = self.s.recv(1024)
            data += buffer
            if len(buffer) == 1024:
                pass
            else:
                return data

    def startStream(self):
        """Decodes files from stream and displays them"""
        img = np.zeros((3, 3)) # make blank img

        # initial frame cant use intra-frame compression
        prevFrame = np.load(io.BytesIO(self.D.decompress(recv_msg(self.s))))
        frameno=0

        try:
            while True:
                r = recv_msg(self.s) #gets the frame difference
                if r is None:
                    return
                if len(r) == 0:
                    continue
    
                # load decompressed image
                try:
                            #np.load creates an array from the serialized data
                    img = (np.load(io.BytesIO(self.D.decompress(r))) #decompress the incoming frame difference
                     + prevFrame).astype("uint8") # add the difference to the previous frame and convert to uint8 for safety
    
                    if self.verbose:
                        self.log("recieved {}KB (frame {})".format(int(len(r)/1000),frameno)) #debugging
                        frameno+=1
    
                except Exception as e:
                    print(e)
                    self.close()
    
                prevFrame = img #save the frame
    
                if self.Write:
                    self.out.write(img) #save frame to a video file client side
    
                # show it scaled up
                cv2.imshow("feed", cv2.resize(
                    img, (0, 0), fx=self.viewScale, fy=self.viewScale))
    
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        finally:
            self.out.release()
            cv2.destroyAllWindows()
            
    def close(self):
        """Closes socket and opencv instances"""
        self.s.close()



#if you directly run this file it will act run this
if __name__ == "__main__": 
    client = Client(serverIp="18.111.87.85", port=5000)
    client.startStream()
