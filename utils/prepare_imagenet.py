import os
import shutil
import sys
import urllib.request

SYNSET_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
)


def organize(val_dir):
    print("Downloading synset labels...")
    response = urllib.request.urlopen(SYNSET_URL)
    synsets = response.read().decode().strip().split("\n")
    print(f"Got {len(synsets)} labels")

    images = sorted(f for f in os.listdir(val_dir) if f.endswith(".JPEG"))
    print(f"Found {len(images)} images in {val_dir}")

    if not images:
        print("No JPEG files found.")
        return

    if len(images) != len(synsets):
        print(f"WARNING: {len(images)} images but {len(synsets)} labels")

    moved = 0
    for img_name, synset in zip(images, synsets):
        dest_dir = os.path.join(val_dir, synset)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(os.path.join(val_dir, img_name), os.path.join(dest_dir, img_name))
        moved += 1

    print(f"Done. Moved {moved} images into class subdirectories.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m utils.organize_imagenet_val <path-to-val-dir>")
        sys.exit(1)

    val_dir = sys.argv[1]
    if not os.path.isdir(val_dir):
        print(f"Error: {val_dir} is not a directory")
        sys.exit(1)

    organize(val_dir)
