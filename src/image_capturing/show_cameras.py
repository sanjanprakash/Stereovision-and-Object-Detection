import argparse
import os
import time

import cv2

from stereovision.stereo_cameras import StereoPair


def main():
    """
    Show the video from two webcams successively.
    For best results, connect the webcams while starting the computer.
    I have noticed that in some cases, if the webcam is not already connected
    when the computer starts, the USB device runs out of memory. Switching the
    camera to another USB port has also caused this problem in my experience.
    """
    parser = argparse.ArgumentParser(description="Show video from two "
                                     "webcams.\n\nPress 'q' to exit.")
    parser.add_argument("devices", type=int, nargs=2, help="Device numbers "
                        "for the cameras that should be accessed in order "
                        " (left, right).")
    parser.add_argument("--output_folder",
                        help="Folder to write output images to.")
    parser.add_argument("--interval", type=float, default=1,
                        help="Interval (s) to take pictures in.")
    args = parser.parse_args()

    with StereoPair(args.devices) as pair:
        if not args.output_folder:
            pair.show_videos()
        else:
            i = 1
            while True:
                start = time.time()
                while time.time() < start + args.interval:
                    pair.show_frames(1)
                images = pair.get_frames()
                for side, image in zip(("left", "right"), images):
                    filename = "{}_{}.png".format(side, i)
                    output_path = os.path.join(args.output_folder, filename)
                    cv2.imwrite(output_path, image)
                i += 1


if __name__ == "__main__":
    main()