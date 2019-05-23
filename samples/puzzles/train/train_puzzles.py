import copy
import image_utils as iu

# Image processing pipeline, data augmentation, etc
# Load annotated images.
annotated_images = iu.load_dict_from_json("images/zones.json")

# Crop images using mask/polygon coordinates and save individual cropped images to file, ex. puzzle_1.png.
cropped_image_dict = iu.crop_masked_objects(annotated_images, "images/zones_1024.jpg", "puzzle",
                                            "images/cropped/via_region_data_cropped_masks.json")

#  Just copy passed in file and reset image metadata content.
new_image_dict = copy.deepcopy(cropped_image_dict)
new_image_dict["_via_img_metadata"] = {}

# Get the keys holding the cropped image names.
keys = cropped_image_dict["_via_img_metadata"].keys()

# Take new cropped images and paste them in random positions and orientations on the background image.
# Since it is not feasible to hand annotate enough training examples automate sample creation.
for sample_number in range(1, 360):
    iu.add_training_sample(sample_number, 'images/background_1024.jpg', cropped_image_dict, new_image_dict,
                           sample_number)

iu.dump_dict_to_json(new_image_dict, 'puzzle_val.json')
print("train puzzles complete...")
