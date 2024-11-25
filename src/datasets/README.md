# Datasets

The following datasets are currently supported:
- [x] MOT17
- [x] DanceTrack
- [x] KITTIMOT
- [x] MOT20

# Custom dataset

To use a custom dataset with a tracker, follow these steps:

## Step 1 - Classes

1) Create a new file here name `<dataset>.py` in this directory
2) Specify the `PATH_DATA`, which should point to the folder containing your dataset
3) Register the new dataset class by adding it to the `__init__.py` file

## Step 2 - Required properties

Ensure the dataset class includes the following properties:

### General properties

- `list_subsets`: list of all the available subsets (e.g. ['train', 'val', 'test'])
- `dict_COI`: dictionary mapping objectIDs to their labels (e.g. {1: 'pedestrian', 3: 'car'})
- `framestep`: step size for frame naming (e.g. 1)
- `level`: valuation precision level (e.g. 'box')
- `list_scenes`: list of all video sequences names (e.g. ['0001', '0002', '0003'])

### Scene-specific properties loaded via `load_scene()`

- `scene_name`: name of the loaded scene (e.g. '0002')
- `height`: height of images (e.g. 1080)
- `width`: width of images (e.g. 1920)
- `fps`: frame per second (e.g. 25)
- `list_frames`: all frames indices in the scene (e.g. [1, 2, 3, 4, 5, 6, 7, 8])

## Step 3 - Required methods

The dataset class must implement the following methods:
- `load_scene(self, scene_name)`: load the information related to a scene
- `load_oracle_infos(self)`: load ground truth annotations (e.g., bounding boxes, objectIDs, ignored areas)
- `get_path_image(self, frame)`: return the path to the image corresponding to a given frame
- `load_benchmarks(self, filename)`: load tracklets from a benchmark file
- `save_benchmark(self, tracklets, **kwargs)`: save tracker results based on frame-level data to a file
- `save_benchmark_from_Track(self, memory, **kwargs)`: save tracker results based on object-level data to a file
- `evaluate_performance(self, **kwargs)`: run the evaluation code to assess tracker performance
