# Graphics Raytracing Project

## Running the project

### Installing requirements

We assume that you already have Python 3+ on your machine. If not, you can follow the instructions for your operating system to set up Python. No tricky packages need to be installed, so you don't have to worry about complex dependencies.

The Python packages required to run this project can be installed via

```shell
pip install -r requirements.txt
```

### Running the script

**Usage**

```shell
# generate frames
python3 ray_trace.py [--width WIDTH] [--height HEIGHT] [--grid GRID]
                     [--duration DURATION] [--fps FPS] [--threads THREADS]
                     [--reflection_depth REFLECTION_DEPTH]
                     [--transmission_depth TRANSMISSION_DEPTH] [--mirrorball]


# write frames into animated gif
python3 export_gif.py
```

**Examples**

```shell
# test render of 13 frames
python3 ray_trace.py --duration 13 --fps 1

# test render with mirror ball
python3 ray_trace.py --mirrorball

# high resolution render with mirror ball
python3 ray_trace.py --mirrorball --width 256 --height 256
```

### Command line options

```shell
  -h, --help            show this help message and exit
  --width WIDTH         Width of output image
  --height HEIGHT       Height of output image
  --grid GRID           Grid size of the wave mesh
  --duration DURATION   Duration of animation to export
  --fps FPS             Frame per second of duration to export
  --threads THREADS     Number of threads to use
  --reflection_depth REFLECTION_DEPTH
                        Maximum recursion depth for reflection
  --transmission_depth TRANSMISSION_DEPTH
                        Maximum recursion depth for transmission
  --mirrorball          Draw the mirror ball in the scene
```
