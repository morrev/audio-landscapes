"""
3D mesh responding to music
"""
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import sys
import pyaudio
import wave
import argparse
from scipy import signal
import contextlib
import os
import matplotlib.cm as cm

SUPPRESS_WARNINGS = True
SCREENDIM = (0, 110, 1920, 1080) #Dimensions of QtGui QApplication window
CAMERA_DISTANCE = 110
CAMERA_ELEVATION = 8
STEPSIZE = 1 #Lower = more granular mesh (more compute)
REFRESH_MS = 10 #Number of milliseconds between refresh: setting too high results in audio buffer underrun
X_MIN = -32; X_MAX = 32
Y_MIN = -32; Y_MAX = 32
VISUALIZER = 'spectrogram'
IGNORE_THRESHOLD = 0.5 #Higher ignore threshold --> plot less audio noise in the 3D mesh
PREV_WEIGHT = 0.3 #Weight to assign to previous observation (to 'smooth' peaks)
TRANSLUCENCY = 0.8 #Translucency of faces in mesh
SCALE = 0.1 #Scale heights: larger = higher

def to_channel_matrix(data, n_channels):
    """Return channels given frames of data"""
    # todo: dynamic dtype? {1:np.int8, 2:np.int16, 4:np.int32}[self.wf.getsampwidth()]
    data = np.frombuffer(data, dtype='b')
    # Skip every nchannel entry in the frame array to extract channels as rows
    channel_matrix = np.stack([data[i::n_channels] for i in range(n_channels)], axis = 0)
    return channel_matrix

def to_mono(channel_matrix, agg_function = np.mean):
    """Aggregate multichannel matrix (from e.g. stereo into mono) using agg_function"""
    agged_matrix = agg_function(channel_matrix, axis = 0)
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
        self.window.setCameraPosition(distance = CAMERA_DISTANCE, elevation = CAMERA_ELEVATION)
        
        # Define the color map (from matplotlib.cm)
        # cmap_agger defines how to aggregate face vertex heights to determine the face color
        self.cmap = cmap
        self.cmap_agger = np.mean

        # Define the visualizer (raw bytes or spectrogram)
        self.visualizer = self._setvisualizer(visualizer)

        # Create the mesh item
        self._setverts()
        self._setfaces()
        self.mesh = gl.GLMeshItem(
            vertexes = self.verts,
            faces = self.faces, faceColors = self.colors,
            smooth=False, drawEdges=True,
        )
        self.mesh.setGLOptions('additive')
        self.window.addItem(self.mesh)
    
        # Initialize audio stream
        self._setaudiostream(audio_filename)

    def _setvisualizer(self, visualizer):
        if visualizer == 'spectrogram':
            return self._get_spectrogram_heights
        elif visualizer == 'bytes':
            return self._get_wav_heights
        else:
            raise NotImplementedError

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
        print(f"Chunk (frames/buffer): {self.chunk}")

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
        self.wf = wave.open(audio_filename, 'rb')
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

    def _get_wav_heights(self, mono):
        """Return proposed mesh heights based on raw mono (single channel) values from wav"""
        # Then sample every other byte to match the 3D grid mesh size (chunk)
        # Add 128 since 'b' datatype supports -128 to 128, and we want positive jumps in the grid mesh
        proposed_heights = (mono[::self.wf.getsampwidth()] + 128) * SCALE
        return proposed_heights

    def _get_spectrogram_heights(self, mono):
        """Return proposed mesh heights based on spectrogram, given mono values from wav"""
        frequencies, times, S = signal.spectrogram(mono, 
            fs = 10, #self.wf.getframerate(),
            nperseg = (2*self.grid_width) - 1,
            noverlap = 20, # higher noverlap leads to smoothing of spectrogram across time axis
            window = 'hann'
        )
        S = fit_to_shape(S, self.grid_width, self.grid_height).flatten()
        proposed_heights = SCALE * np.log(S)
        assert(len(proposed_heights)==self.chunk)
        return proposed_heights

    def _get_colors(self, proposed_heights):
        """Given an array of proposed heights, return proposed colors for faces"""
        # Get heights of each vertex defining each face, for each face
        vertex_heights = proposed_heights[self.faces] 
        face_color_idx = self.cmap_agger(vertex_heights, axis = 1)
        return self.cmap(face_color_idx)

    def update(self):
        """Update the mesh heights and play audio stream"""
        # Every read of data contains chunk * sample_width * n_channels bytes
        data = self.wf.readframes(self.chunk)

        # If audio file hasn't ended (i.e. not all chunks have been read)...
        if data:
            channel_matrix = to_channel_matrix(data, n_channels = self.wf.getnchannels())
            mono = to_mono(channel_matrix, np.sum)  

            proposed_heights = self.visualizer(mono)

            # Crude denoising
            proposed_heights[proposed_heights < IGNORE_THRESHOLD] = 0

            # Set mesh heights
            new_heights = self.verts[:,2]*PREV_WEIGHT + proposed_heights*(1-PREV_WEIGHT) # weight previous height
            self.verts[:,2] = new_heights
            
            # Play audio sound
            self.stream.write(data)
            
            # Set face colors
            new_face_colors = self._get_colors(new_heights)

            # Update mesh heights
            self.mesh.setMeshData(
                vertexes = self.verts, 
                faces = self.faces, 
                faceColors = new_face_colors
            )
        
        # Prevent underruns by filling with silence if have more than chunksize free space in the buffer
        free = self.stream.get_write_available()
        if free > self.chunk:
            self.stream.write(chr(0) * (free - self.chunk))

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
        t = Terrain(audio_filename = args.audio_filename, visualizer = VISUALIZER, cmap = cm.afmhot, cmap_agger = np.mean)
        t.animation()
