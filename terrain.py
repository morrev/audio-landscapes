"""
3D mesh responding to music
"""
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import sys
import pyaudio
import wave
import soundfile as sf
import argparse
from scipy import signal
import contextlib
import os
import matplotlib.cm as cm
from skimage.transform import resize

SUPPRESS_WARNINGS = True
SCREENDIM = (0, 110, 1920, 1080) #Dimensions of QtGui QApplication window

CAMERA_DISTANCE = 100
CAMERA_ELEVATION = 30
CAMERA_ROTATION = 0

STEPSIZE = 1 #Lower = more granular mesh (more compute)
IGNORE_THRESHOLD = 2 #Higher ignore threshold --> plot less audio noise in the 3D mesh
TRANSLUCENCY = 0.5 #Translucency of faces in mesh
COLOR_SCALE = 0.2 #Scale colors
SCALE = 0.8 #Scale heights: larger = higher
CMAP = cm.cool #Colormap for the faces
CMAP_AGGER = np.mean #How to aggregate face vertex heights to determine the face color
DRAW_EDGES = False

WINDOW_SIZE = 4096
WINDOW = np.hamming(WINDOW_SIZE)
HOP_LENGTH = int(WINDOW_SIZE * (3/4))
OVERLAP = WINDOW_SIZE - HOP_LENGTH
REFRESH_MS = 5 #Number of milliseconds between refresh: setting too high results in audio buffer underrun
print(f"Refresh (ms): {REFRESH_MS}")

Y_HEIGHT = 80
X_HEIGHT = 80

MIN_FREQUENCY = 20
MAX_FREQUENCY = 1000 # Max Hz to display
X_MIN = -(X_HEIGHT//2); X_MAX = (X_HEIGHT//2) # time
Y_MIN = -(Y_HEIGHT//2); Y_MAX = (Y_HEIGHT//2) # frequency

def to_mono(channel_matrix, agg_function = np.mean):
    """Aggregate multichannel matrix (from e.g. stereo into mono) using agg_function"""
    agged_matrix = agg_function(channel_matrix, axis = 1)
    return agged_matrix

def fit_to_shape(matrix, height, width):
    """
    Pad and truncate numpy matrix to shape (e.g. for plotting spectrogram)
    Example usage: fit_to_shape(S, self.grid_height, self.grid_width)
    """
    matrix = matrix[:height,:width]
    h_pad = max([0, width - matrix.shape[1]])
    v_pad = max([0, height - matrix.shape[0]])
    padded_matrix = np.pad(matrix, ((0,v_pad),(0,h_pad)))
    return padded_matrix

@contextlib.contextmanager
def ignore_stderr():
    """Suppress pyaudio stderr warning messages"""
    # https://stackoverflow.com/questions/36956083/how-can-the-terminal-output-of-executables-run-by-python-functions-be-silenced-i/36966379#36966379
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

class Terrain(object):
    def __init__(self, audio_filename, visualizer = 'spectrogram', cmap = cm.autumn, cmap_agger = np.mean):
        """Initialize the graphics window, mesh, and audio stream"""
        # Set up the view window
        self.app = QtGui.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setGeometry(*SCREENDIM)
        self.window.show()
        self.window.setWindowTitle('Terrain')
        self.window.setCameraPosition(distance = CAMERA_DISTANCE, 
            elevation = CAMERA_ELEVATION,
            azimuth = CAMERA_ROTATION)
        
        # Define the color map (from matplotlib.cm)
        # cmap_agger defines how to aggregate face vertex heights to determine the face color
        self.cmap = cmap
        self.cmap_agger = np.mean

        # Create the mesh item
        self._setverts()
        self._setfaces()
        self.mesh = gl.GLMeshItem(
            vertexes = self.verts,
            faces = self.faces, faceColors = self.colors,
            smooth=False, drawEdges=DRAW_EDGES,
        )
        self.mesh.setGLOptions('additive')
        self.window.addItem(self.mesh)
    
        # Initialize audio stream
        self._setaudiostream(audio_filename)

    def _setverts(self):
        """Create array of vertices"""
        xx = np.arange(X_MIN, X_MAX, STEPSIZE)
        yy = np.arange(Y_MIN, Y_MAX, STEPSIZE)
        self.verts = np.array([
            [x, y, 0] for x in xx for y in yy
        ], dtype=np.float32)

        # Chunk is the number of points in the grid: determines # frames/buffer (i.e. coincident audio samples per buffer) 
        self.chunk = len(self.verts)
        self.grid_width = len(xx); self.grid_height = len(yy)
        print(f"Number of vertices: {self.chunk}")

    def _setfaces(self):
        """Create triangular faces"""
        faces = []
        for y in range(self.grid_width - 1):
            yoff = y * self.grid_height
            for x in range(self.grid_height - 1):
                faces.append([x + yoff, x + yoff + self.grid_height, x + yoff + self.grid_height + 1])
                faces.append([x + yoff, x + yoff + 1, x + yoff + self.grid_height + 1])
        self.faces = np.array(faces)
        self.colors = self._get_colors(np.zeros(self.chunk))
        
    def _setaudiostream(self, audio_filename):
        """Set audio stream"""
        # Obtain audio metadata by opening with wave (e.g. sample width, number channels, etc.)
        self.sf = wave.open(audio_filename, 'rb')
        self.n_channels = self.sf.getnchannels() # Number of channels per frame
        self.sr = self.sf.getframerate() # Sampling rate: # frames (samples) / second
        self.samp_width = self.sf.getsampwidth() # Sample width in bytes

        # get sample frequencies associated with rfft
        freq = np.fft.rfftfreq(WINDOW_SIZE, d = 1./self.sr)
        self.min_freq_index = np.argmax(freq > MIN_FREQUENCY)
        self.max_freq_index = np.argmax(freq > MAX_FREQUENCY)

        # Use soundfile to open the file to obtain a generator passing audio blocks with overlap
        self.wf = sf.blocks(audio_filename, blocksize = WINDOW_SIZE, overlap = OVERLAP)
        
        # Create stream with pyaudio to allow playing the audio sound
        p = pyaudio.PyAudio()
        self.stream = p.open(format = p.get_format_from_width(self.samp_width),
                channels = self.n_channels,
                rate = self.sr,
                input = False, 
                output = True)
        print(f"Bytes per sample: {self.samp_width}")
        print(f"Channels: {self.n_channels}")
        print(f"Rate: {self.sr}")

    def _get_colors(self, proposed_heights):
        """Given an array of proposed heights, return proposed colors for faces"""
        # Get heights of each vertex defining each face, for each face
        vertex_heights = proposed_heights[self.faces] 
        face_color_idx = self.cmap_agger(vertex_heights, axis = 1)
        return self.cmap(face_color_idx)

    def update(self):
        """Update the mesh heights and play audio stream"""
        data = next(self.wf) # Get next window of the waveform, of shape (WINDOW_SIZE, n_channels)
        
        # If audio file hasn't ended (i.e. not all chunks have been read)...
        if len(data) > 0:
            mono = to_mono(data, np.mean)  

            ft = np.fft.rfft(WINDOW * mono)[self.min_freq_index:self.max_freq_index]
            ft = np.sqrt(np.abs(ft))
            ft = resize(ft.reshape(-1,1), (Y_HEIGHT, 1)).squeeze()
            ft[ft < IGNORE_THRESHOLD] = 0.0
            
            self.verts[:,2][:self.grid_height] = ft
            self.verts[:,2][self.grid_height:] = self.verts[:-self.grid_height, 2]

            # Play audio sound
            self.stream.write(self.sf.readframes(HOP_LENGTH))
            
            # Set face colors
            new_face_colors = self._get_colors(self.verts[:,2] * COLOR_SCALE)

            self.verts[:,2][:self.grid_height] *= SCALE

            # Update mesh heights
            self.mesh.setMeshData(
                vertexes = self.verts, 
                faces = self.faces, 
                faceColors = new_face_colors
            )
        
        # Prevent underruns by filling with silence if have more than chunksize free space in the buffer
        free = self.stream.get_write_available()
        if free > WINDOW_SIZE:
            self.stream.write(chr(0) * (free - WINDOW_SIZE))

    def start(self):
        """Open graphics window"""
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec()

    def animation(self):
        """Call update to refresh plot (and play audio chunk)"""
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(REFRESH_MS) #time in milliseconds between each call to method
        self.start()
        self.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--audio_filename", required = True, help = "Enter .wav filepath")
    args = parser.parse_args()
    
    # Suppress pyaudio stderr messages
    with (ignore_stderr() if SUPPRESS_WARNINGS else contextlib.nullcontext()) as silence:
        t = Terrain(audio_filename = args.audio_filename, cmap = CMAP, cmap_agger = CMAP_AGGER)
        t.animation()
