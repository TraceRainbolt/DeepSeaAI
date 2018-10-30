import csv
import os

from shutil import copy

from PIL import Image

new_dirs = "seaCreatures"
crop_dirs = "cropSeaCreatures"

path = os.getcwd()

new_path = os.path.join(path, new_dirs)
crop_path = os.path.join(path, crop_dirs)


def make_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def crop(image_path, coords, saved_location, image_name):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save("{}\{}".format(saved_location, image_name))
    cropped_image.show()


with open('annotationData.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for i, row in enumerate(csv_reader):
        if i % 2 == 0 and i != 0:
            image_name = row[5]
            class_ = row[7]

            if class_ == "Unknown":
                continue

            coords = [float(x) for x in row[1:5]]

            print(f'\tImgId: {row[5]} \t Classification: {class_}')

            if image_name[-3:] != "png":
                image_name += ".png"

            final_path = new_path + "\{}".format(class_)
            final_crop_path = crop_path + "\{}".format(class_)

            make_dir(final_path)
            make_dir(final_crop_path)

            crop(r"deepSeaData\\annotations\\{}".format(image_name), coords, final_crop_path, image_name)
