import imageio
import glob

filenames = [f.zfill(7) for f in glob.glob("*.png")]
filenames.sort()

print("Writing gif...")

with imageio.get_writer("water.gif", mode="I", duration=0.1) as writer:
    for filename in filenames:
        filename = filename.lstrip("0")
        if filename == ".png":
            filename = "0.png"
        image = imageio.imread(filename)
        writer.append_data(image)

print("Done!")
