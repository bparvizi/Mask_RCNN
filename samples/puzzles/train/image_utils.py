import os
import json
import copy
import cv2
import datetime
import numpy as np
import random
from PIL import Image, ImageDraw


def load_dict_from_json(file_path):
    with open(file_path, 'r') as f:
        json_dict = json.load(f)
    return json_dict


def dump_dict_to_json(json_dict, file_path):
    json.dump(json_dict, open(file_path, 'w', encoding='utf-8'))


def crop_transparent_object(image_path, x_coords, y_coords, out_file):
    # Get min and max values required for creating offsets and new width and height for cropped image.
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Used for creating new images with cropped dimensions.
    x_length = x_max - x_min
    y_length = y_max - y_min

    # Offset coordinates.
    x_coords_offset = np.array(x_coords) - x_min
    y_coords_offset = np.array(y_coords) - y_min

    zipped = tuple(zip(x_coords, y_coords))
    polygon = list(zipped)

    # Read image as RGB and add alpha (transparency)
    full_image = np.asarray(Image.open(image_path).convert("RGBA"))

    # Create masked image using polygon region.
    masked_image = Image.new('L', (full_image.shape[1], full_image.shape[0]), 255)
    ImageDraw.Draw(masked_image).polygon(polygon, outline=1, fill=1)
    masked_image_array = np.array(masked_image)
    # masked_image.save('masked_image.png')

    # Assemble new image (uint8: 0-255)
    newImArray = np.empty(full_image.shape, dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:, :, :3] = full_image[:, :, :3]

    # transparency (4th column)
    newImArray[:, :, 3] = masked_image_array * 255

    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")

    newImArrayPaste = np.empty((y_length, x_length, 4), dtype='uint8')
    newImPaste = Image.fromarray(newImArrayPaste, "RGBA")

    cropped_image = newIm.crop((x_min, y_min, x_max, y_max))
    newIm.paste(cropped_image)

    newImPaste.paste(cropped_image)
    newImPaste.save(out_file)

    return x_coords_offset.tolist(), y_coords_offset.tolist()


def get_annotation(filename, size, all_points_x, all_points_y, class_id, empty_regions=False):
    annotation = {
        "filename": filename,
        "size": size,
        "regions": [
            {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y,
                    "class_id": class_id
                },
                "region_attributes": {}
            },
        ],
        "file_attributes": {}}

    if empty_regions:
        annotation["regions"] = []

    return annotation


def crop_masked_objects(json_dict, image_file_path, file_name_prefix, out_file_path):
    # Get all the keys.
    keys = json_dict["_via_img_metadata"].keys()

    # Create new JSON dictionary for cropped images used to inspect cropped results.
    cropped_image_dict = copy.deepcopy(json_dict)
    # Just copy passed in file and reset image metadata content.
    cropped_image_dict["_via_img_metadata"] = {}

    # Parse all regions for each key.
    for key in keys:
        # Each region corresponds to a different class.
        regions = json_dict["_via_img_metadata"][key]["regions"]
        for i, region in enumerate(regions):
            # Assign class id.
            class_id = i + 1
            cropped_file_name = "images/cropped/" + file_name_prefix + "_" + str(class_id) + ".png"
            x = region["shape_attributes"]["all_points_x"]
            y = region["shape_attributes"]["all_points_y"]

            # Crop each masked object and create new annotation entry.
            x_coords_offset, y_coords_offset = crop_transparent_object(image_file_path, x, y, cropped_file_name)

            # Get image size required to key new annotation.
            file_size = os.path.getsize(cropped_file_name)

            # Create new annotation.
            annotation = get_annotation(cropped_file_name, file_size, x_coords_offset, y_coords_offset, class_id)

            # Add new record to JSON.
            new_record = cropped_file_name + str(file_size)
            cropped_image_dict["_via_img_metadata"][new_record] = annotation

    json.dump(cropped_image_dict, open(out_file_path, 'w', encoding='utf-8'))
    return cropped_image_dict


def save_movie_frame(video_path, save_file_name):
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # Define codec and create video writer
    file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              fps, (width, height))
    success, image = vcapture.read()
    print(success)
    if success:
        # OpenCV returns images as BGR, convert to RGB
        image = image[..., ::-1]
        pill_image = Image.fromarray(image)
        pill_image.thumbnail((1024, 1024), Image.ANTIALIAS)
        pill_image.save(save_file_name, "JPEG")


def get_random_angle():
    # angles = [33, 120, 176, 290]
    # random_index = random.randint(0, 3)
    # return angles[random_index]
    return random.randint(1, 360)


def shuffle_positions():
    positions = [(10, 10), (10, 175), (10, 350), (215, 10), (215, 175), (215, 350), (490, 250), (490, 50)]
    random.shuffle(positions)
    return positions


def create_rotated_image(image, degrees, file):
    """
    Parameters
    ----------
    image : ndarray
        PIL image
    degrees : int
        Rotation in degrees.
    file : str
        Name of the file to rotate.
    """
    file_name, ext = file.split(".")

    # Rotate image.
    rotated_image = image.rotate(degrees, expand=True)

    # The expand=True arg from previous step will resize the image.
    # Take the max width or height and resize to thumbnail.
    max_length = max(rotated_image.size)

    if max_length > 190:
        max_length = 190

    # Find the new image center using the original dimensions against the new rotated one.
    center_x = int(np.ceil((max_length - image.size[0]) / 2))
    center_y = int(np.ceil((max_length - image.size[1]) / 2))

    # Create a new transparent background with new dimensions.
    image_array = np.empty((max_length, max_length, 4))
    transparent_image = Image.fromarray(image_array, "RGBA")

    # Paste original image in container.
    transparent_image.paste(image, (center_x, center_y), mask=image)

    # Rotate transparent image.
    transparent_image = transparent_image.rotate(degrees)

    # Save.
    save_file = file_name + '_test_' + str(degrees) + '.' + ext
    # transparent_image.save(save_file)

    return save_file, center_x, center_y, transparent_image


def rotate_segmentation_mask(center_x, center_y, transparent_image, degrees, x_coords, y_coords):
    # Update coordinates with new center after image was resized.
    x_coords = np.array(x_coords) + center_x
    y_coords = np.array(y_coords) + center_y

    # Find center, skimage.transform centers using: center=(cols / 2 - 0.5, rows / 2 - 0.5)
    center_x = transparent_image.size[1] / 2
    center_y = transparent_image.size[0] / 2

    # Calculate cos/sin functions.
    theta = np.radians(degrees)  # Convert angle to radians
    cos, sin = np.cos(theta), np.sin(theta)

    # Find rotated values.
    x_coords_rotated = []
    y_coords_rotated = []
    for i in range(x_coords.shape[0]):
        x, y = x_coords[i], y_coords[i]
        tx, ty = x - center_x, y - center_y
        new_x = np.int(np.round((tx * cos + ty * sin) + center_x))
        new_y = np.int(np.round((-tx * sin + ty * cos) + center_y))
        x_coords_rotated.append(new_x)
        y_coords_rotated.append(new_y)

    return x_coords_rotated, y_coords_rotated


def add_training_sample(sample_number, background_image, cropped_image_dict, new_image_dict):
    # Load background image.
    background = Image.open(background_image)

    save_file_name = "puzzle_train_" + str(sample_number) + ".png"

    # Hold the new regions to be added to annotation file.
    pasted_regions = []

    # Paste images on background in variety of positions and angles.
    positions = shuffle_positions()

    # Get keys.
    keys = cropped_image_dict["_via_img_metadata"].keys()
    for i, key in enumerate(keys):
        region = cropped_image_dict["_via_img_metadata"][key]["regions"][0]
        region_copy = copy.deepcopy(region)

        # Class Id.
        class_id = region["shape_attributes"]["class_id"]

        # Load the initial image.
        file_name = "images/cropped/puzzle_" + str(class_id) + '.png'
        loaded_image = Image.open(file_name).convert("RGBA")

        # Create rotated image and save to file.
        rotation_degrees = get_random_angle()
        rotated_image_file_path, center_x, center_y, rotated_image = create_rotated_image(loaded_image,
                                                                                          rotation_degrees, file_name)
        # Get position
        position = positions[i]

        # Paste image on background.
        background.paste(rotated_image, position, mask=rotated_image)

        # Get x and y coordinates.
        x = np.array(region["shape_attributes"]["all_points_x"])
        y = np.array(region["shape_attributes"]["all_points_y"])

        # Adjust x and y mask coordinates.
        x_coords_rotated, y_coords_rotated = rotate_segmentation_mask(center_x, center_y, rotated_image,
                                                                      rotation_degrees, x, y)

        # Updated coordinates to new position.
        x_coords_rotated = np.array(x_coords_rotated) + position[0]
        y_coords_rotated = np.array(y_coords_rotated) + position[1]

        # Set new coordinates.
        region_copy["shape_attributes"]["all_points_x"] = x_coords_rotated.tolist()
        region_copy["shape_attributes"]["all_points_y"] = y_coords_rotated.tolist()
        pasted_regions.append(region_copy)

    # Save file
    background.save(save_file_name)

    # Get file size
    file_size = os.path.getsize(save_file_name)
    print("file_size: ", file_size)

    # Create new annotation.
    annotation = get_annotation(save_file_name, file_size, '', '', '', empty_regions=True)
    annotation["regions"] = pasted_regions

    # Add new record to JSON.
    new_record = save_file_name + str(file_size)

    new_image_dict["_via_img_metadata"][new_record] = annotation
