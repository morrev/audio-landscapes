"""
3D mesh responding to music
"""
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import struct
import sys
import librosa
import pyaudio
import wave

SCREENDIM = (0, 110, 1920, 1080)
CAMERA_DISTANCE = 70
CAMERA_ELEVATION = 8
STEPSIZE = 0.8 #Lower = more granular mesh (more compute)
REFRESH_MS = 2 #Number of milliseconds between refresh: setting too high results in audio buffer underrun
X_MIN = -16; X_MAX = 16
Y_MIN = -16; Y_MAX = 16
#AUDIO_FILE = './data/Bach_Canon_2_from_Musical_offering.wav' 
AUDIO_FILE = './data/Chopin_op28_excerpt.wav'
#AUDIO_FILE = './data/francois_couperin.wav'
IGNORE_THRESHOLD = 0.5 #Higher ignore threshold --> plot less audio noise in the 3D mesh
PREV_WEIGHT = 0.5 #Weight to assign to previous observation (to 'smooth' peaks)
TRANSLUCENCY = 0.9 #Translucency of faces in mesh
SCALE_DOWN = 0.005 #Weight to scale down the heights (smaller = less high)

class Terrain(object):
    def _setverts(self):
        """Create array of vertices"""
        xx = np.arange(X_MIN, X_MAX, STEPSIZE)
        yy = np.arange(Y_MIN, Y_MAX, STEPSIZE)
        self.verts = np.array([
            [x, y, 0] for x in xx for y in yy
        ], dtype=np.float32)

        # Chunk is the number of points in the grid: determines # frames/buffer (i.e. coincident audio samples per buffer) 
        self.chunk = self.verts.shape[0]
        self.grid_width = len(xx); self.grid_height = len(yy)
        print(f"Chunk (frames/buffer): {self.chunk}")

    def _setfaces(self):
        """Create triangular faces"""
        faces = []
        colors = []
        for y in range(self.grid_width - 1):
            yoff = y * self.grid_width
            for x in range(self.grid_width - 1):
                faces.append([x + yoff, x + yoff + self.grid_width, x + yoff + self.grid_width + 1])
                faces.append([x + yoff, x + yoff + 1, x + yoff + 1 + self.grid_width])
                colors.append([x / self.grid_width, 1 - x / self.grid_width, y / self.grid_width, TRANSLUCENCY-0.1])
                colors.append([x / self.grid_width, 1 - x / self.grid_width, y / self.grid_width, TRANSLUCENCY])
        self.faces = np.array(faces)
        self.colors = np.array(colors)
        print(self.faces.shape)

    def _setaudiostream(self):
        """Set audio stream"""
        self.wf = wave.open(AUDIO_FILE, 'rb')
        p = pyaudio.PyAudio()
        
        self.stream = p.open(format = p.get_format_from_width(self.wf.getsampwidth()),
                channels = self.wf.getnchannels(), #Number of channels per frame
                rate = self.wf.getframerate(), #Sampling rate: # frames / second
                input = False, 
                output = True)
        print(f"Channels: {self.wf.getnchannels()}")
        print(f"Rate: {self.wf.getframerate()}")
        print(f"Bytes per sample: {self.wf.getsampwidth()}")
        
        # Number of bytes in each update is: (# channels) * (bytes/sample) * (chunk)
        self.num_bytes = self.wf.getnchannels() * self.wf.getsampwidth() * self.chunk
        print(f"Bytes per update: {self.num_bytes}")

    def __init__(self):
        """Initialize the graphics window, mesh, and audio stream"""
        # Set up the view window
        self.app = QtGui.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setGeometry(*SCREENDIM)
        self.window.show()
        self.window.setWindowTitle('Terrain')
        self.window.setCameraPosition(distance = CAMERA_DISTANCE, elevation = CAMERA_ELEVATION)
        
        self._setverts()
        self._setfaces()
        
        # Create the mesh item
        self.mesh = gl.GLMeshItem(
            vertexes = self.verts,
            faces = self.faces, faceColors = self.colors,
            smooth=False, drawEdges=True,
        )
        self.mesh.setGLOptions('additive')
        self.window.addItem(self.mesh)
    
        # Initialize audio stream
        self._setaudiostream()

    def update(self):
        """Update the mesh heights with audio stream"""
        data = self.wf.readframes(self.chunk)
        self.stream.write(data)
        data = np.array(struct.unpack(str(self.num_bytes) + 'B', data), dtype = 'b')
       
        # Skip every nchannel entry in the bytes array to extract a single channel
        # Then sample every other byte to match the 3D grid mesh size (chunk)
        # Add 128 since 'b' datatype supports -128 to 128, and we want positive jumps in the grid mesh
        channel_zero = data[0::self.wf.getnchannels()]
        channel_one = data[1::self.wf.getnchannels()]
        
        channel = ((channel_zero + channel_one)[::self.wf.getsampwidth()] + 128) * SCALE_DOWN
        #channel_zero = (data[0::self.wf.getnchannels()][::self.wf.getsampwidth()] + 128) * SCALE_DOWN

        # Crude denoising
        channel[channel < IGNORE_THRESHOLD] = 0

        # Set mesh heights
        new_height = self.verts[:,2]*PREV_WEIGHT + (channel * 2)*(1-PREV_WEIGHT) # weight previous height
        self.verts[:,2] = new_height

        self.mesh.setMeshData(
            vertexes=self.verts, faces=self.faces, faceColors=self.colors
        )

    def start(self):
        """
        get the graphics window open and setup
        """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec()

    def animation(self):
        """
        calls the update method to run in a loop
        """
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(REFRESH_MS) #time in milliseconds between each call to method
        self.start()
        self.update()

if __name__ == '__main__':
    #print("Preparing audio data...")
    #y, sample_rate = librosa.load('data/Toccata_et_Fugue_BWV565.ogg')
    #S = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    #S_dB = librosa.power_to_db(S, ref=np.max)
    #print("Done")
    #print(S_dB.shape)
    t = Terrain()
    t.animation()
