Audio landscapes
==========

3D visualization of audio files (currently .wav only).

Computes FFT on windows of .wav chunks to define vertex heights on 3D triangle mesh, with customizable color map and other options.
X axis is time; y axis is frequency.

Hobby project for:
* familiarization w/ audio formats (buffering, channels, underruns)
* exploration of 3D visualization options

Examples:
----
```python terrain.py -f data/audio_file.wav```
![Example](/images/cmap_hot_example.png)
![Example](/images/cmap_bone_example.png)

Visualizer options:
----

* **audio_filename**: filepath of .wav file to play (and plot)
* **cmap**: matplotlib colormap to use for plotting [(options)](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
* **cmap_agger**: method to use to reduce a set of three vertices (defining a face) into a single value (by default, np.mean)
* **refresh_ms**: time between each plot refresh, in milliseconds
* **ignore_threshold**: rounds down to zero for any values below threshold (crude 'denoising')
* *other parameters defined in terrain.py*

Useful links:
----
* [Audio buffer explanation](https://techpubs.jurassic.nl/manuals/0650/developer/DMSDK_PG/sgi_html/ch08.html)
* [PyAudio docs](http://people.csail.mit.edu/hubert/pyaudio/docs/)

Attribution/credits:
----
* Baseline 3D mesh code from [Audio-Spectrum-Analyzer-in-Python](https://github.com/markjay4k/Audio-Spectrum-Analyzer-in-Python): heavily refactored for efficiency and modified to accept .wav files and allow customization/parameters
