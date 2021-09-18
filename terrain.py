"""
3D mesh with noise to simulate a terrain
"""
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import sys

SCREENDIM = (0, 110, 1920, 1080)
CAMERA_DISTANCE = 70
CAMERA_ELEVATION = 8
NSTEPS = 1
NOISE = 0.8
REFRESH_MS = 20 #Number of milliseconds between refresh

class Terrain(object):
    def _setverts(self):
        """Create array of vertices"""
        xx = range(-20, 21, NSTEPS)
        yy = range(-20, 21, NSTEPS)
        self.verts = np.array([
            [x, y, 0] for x in xx for y in yy
        ], dtype=np.float32)
        self.nfaces = len(xx)

    def _setfaces(self):
        """Create triangular faces"""
        faces = []
        colors = []
        for m in range(self.nfaces - 1):
            yoff = m * self.nfaces
            for n in range(self.nfaces - 1):
                faces.append([n + yoff, yoff + n + self.nfaces, yoff + n + self.nfaces + 1])
                faces.append([n + yoff, yoff + n + 1, yoff + n + self.nfaces + 1])
                colors.append([n / self.nfaces, 1 - n / self.nfaces, m / self.nfaces, 0.7])
                colors.append([n / self.nfaces, 1 - n / self.nfaces, m / self.nfaces, 0.8])
        self.faces = np.array(faces)
        self.colors = np.array(colors)
    
    def __init__(self):
        """
        Initialize the graphics window and mesh
        """

        # setup the view window
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.setGeometry(*SCREENDIM)
        self.w.show()
        self.w.setWindowTitle('Terrain')
        self.w.setCameraPosition(distance = CAMERA_DISTANCE, elevation = CAMERA_ELEVATION)
        
        self._setverts()
        self._setfaces()
        
        # create the mesh item
        self.m1 = gl.GLMeshItem(
            vertexes = self.verts,
            faces = self.faces, faceColors = self.colors,
            smooth=False, drawEdges=True,
        )
        self.m1.setGLOptions('additive')
        self.w.addItem(self.m1)

    def update(self):
        """
        update the mesh and shift the noise each time
        """
        self.verts[:,2] = np.random.normal(loc = 0, scale = NOISE, size = self.verts.shape[0])
        self.m1.setMeshData(
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
    t = Terrain()
    t.animation()
