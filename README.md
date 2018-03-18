
**WORK IN PROGRESS**

Idea is to create tools (API, CLI) to store images, targets from a dataset as a few large images to observe the dataset 
in few shots.

## How to use

### Classification dataset

```python

img_files = [
    '/path/to/img_1.ext1',
    # ...
    '/path/to/img_2.ext2',
    # ...
]

targets = [
    'label_0',
    # ...
    12,
    # ...
]

def img_to_RGB_8b(img):
    # Special processing
    return img

def target_to_str(y):
    # Special processing
    return str(y)

from image_dataset_viz import DatasetExporter


de = DatasetExporter(render_img_fn=img_to_RGB_8b, 
                     render_target_fn=target_to_str)
                     
de.export_datapoint(img_files[1], targets[1], "test.png")

output_folder="viz"
de.export(img_files, targets, output_folder, filename_prefix="all_images")
```