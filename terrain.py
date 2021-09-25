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
from scipy import signal

SCREENDIM = (0, 110, 1920, 1080)
CAMERA_DISTANCE = 70
CAMERA_ELEVATION = 8
STEPSIZE = 0.8 #Lower = more granular mesh (more compute)
REFRESH_MS = 4 #Number of milliseconds between refresh: setting too high results in audio buffer underrun
X_MIN = -16; X_MAX = 16
Y_MIN = -16; Y_MAX = 16
#AUDIO_FILE = './data/Bach_Canon_2_from_Musical_offering.wav' 
AUDIO_FILE = './data/Chopin_op28_excerpt.wav'
#AUDIO_FILE = './data/francois_couperin.wav'
IGNORE_THRESHOLD = 1.1 #Higher ignore threshold --> plot less audio noise in the 3D mesh
PREV_WEIGHT = 0.5 #Weight to assign to previous observation (to 'smooth' peaks)
TRANSLUCENCY = 0.9 #Translucency of faces in mesh
SCALE = 0.008 #Weight to scale the heights (smaller = shallower mesh)
    
def to_channel_matrix(data, n_channels):
    """Return channels given frames of data"""
    # todo: dynamic dtype? {1:np.int8, 2:np.int16, 4:np.int32}[self.wf.getsampwidth()]
    data = np.frombuffer(data, dtype='b')
    # Skip every nchannel entry in the frame array to extract channels as rows
    channel_matrix = np.stack([data[i::n_channels] for i in range(n_channels)], axis = 0)
    return channel_matrix

def to_mono(channel_matrix, agg_function = np.mean):
    """Aggregate multichannel matrix (from e.g. into mono"""
    agged_matrix = agg_function(channel_matrix, axis = 0)
    return agged_matrix

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

    def _get_wav_heights(self, mono):
        """Return proposed mesh heights based on raw mono (single channel) values from wav"""
        # Then sample every other byte to match the 3D grid mesh size (chunk)
        # Add 128 since 'b' datatype supports -128 to 128, and we want positive jumps in the grid mesh
        proposed_heights = (mono[::self.wf.getsampwidth()] + 128) * SCALE
        return proposed_heights

    def _get_spectrogram_heights(self, mono):
        """Return proposed mesh heights based on spectrogram, given mono values from wav"""
        frequencies, times, spectrogram = signal.spectrogram(mono) 
        h_pad = self.chunk - frequencies.shape[0]
        proposed_heights = self.chunk - np.pad(frequencies, (0,h_pad))
        return proposed_heights

    def update(self):
        """Update the mesh heights with audio stream"""
        # Every read of data contains chunk * sample_width * n_channels bytes
        data = self.wf.readframes(self.chunk)
        channel_matrix = to_channel_matrix(data, n_channels = self.wf.getnchannels())
        mono = to_mono(channel_matrix, np.sum)  

        #proposed_heights = self._get_wav_heights(mono)
        proposed_heights = self._get_spectrogram_heights(mono)

        # Crude denoising
        proposed_heights[proposed_heights < IGNORE_THRESHOLD] = 0

        # Set mesh heights
        new_heights = self.verts[:,2]*PREV_WEIGHT + proposed_heights*(1-PREV_WEIGHT) # weight previous height
        self.verts[:,2] = new_heights
        
        # Play audio sound
        self.stream.write(data)
        
        # Update mesh heights
        self.mesh.setMeshData(
            vertexes=self.verts, 
            faces=self.faces, 
            faceColors=self.colors
        )
        
        # Prevent underruns by filling with silence
        free = self.stream.get_write_available()
        if free > self.chunk: # Play silence if more free space in buffer than the chunk
            self.stream.write(chr(0) * (free - self.chunk))

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
