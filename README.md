# Object Tracking with Counting and Heatmap

## Introduction

This project offers an advanced solution for real-time multi-object tracking, counting, and activity heatmap generation using state-of-the-art detection and tracking algorithms.

## Features

- Real-time object tracking and counting within a specified Region of Interest (ROI)
- Generation of heatmaps to visualize object activity and movement patterns
- Support for multiple tracking algorithms: DeepSORT, ByteTrack, and more

  

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Repo Overview](#repo-overview)
- [Installation](#installation)
- [Usage](#usage)

## Repo Overview

### boxmot
Contains all the tracking files and corresponding configuration files.

### examples

- **`track.py`**: The origin file of this project, initializing the YOLO model to perform tracking. Arguments are taken using a parser.
  
- **`predictor.py`**: Defines `CustomPredictor` to add new functions and modify the `stream_inference()` function for the counting algorithm and heatmap implementation.
  
- **`YOLO.py`**: Introduces `newYOLO`, a subclass of ultralytics's `YOLO` for task mapping, focusing on detection tasks. Maps `predictor` to `newDetectionPredictor`
  
- **`DetectionPredictor.py`**: Subclass of `CustomPredictor` for postprocessing predictions.
  
- **`Heatmap.py`**: Adds heatmap inference to the output stream with `newHeatmap` subclass.

## Installation


Start with [**Python>=3.8**](https://www.python.org/) environment.

#### step 1: Clone the repository;
```bash 
git clone https://github.com/Project-IPS/CV-Project.git    
cd CV-Project
```

#### step 2: Install requirements;
```bash
pip install -v -e .
```

#### step 3: Setup websocket server;
```bash
cd examples/server
nodemon index.js
```

#### step 4: Run below bash snippet;

```bash
python examples/track.py --yolo-model yolov8.pt --tracking-method botsort --source vid.mp4 --show
```

## Usage

This section provides instructions on how to use the object tracking application with different tracking methods and sources.

### Tracking Methods

The application supports various tracking methods, which can be selected by specifying the `--tracking-method` argument when running `track.py`. Use the following command template, replacing `[method_name]` with the desired tracking method:

```bash
python examples/track.py --tracking-method [method_name]
```
Supported tracking methods include:

* `deepocsort`
* `strongsort`
* `ocsort`
* `bytetrack`
* `botsort`

### Sources
The tracker is capable of processing inputs from various sources, including video formats, images, directories, or live streams. Specify your input source with the `--source` argument as shown below:

```bash
 python examples/track.py --source [input_source]
```


Examples of input sources:

* Webcam: `--source 0`
* Video file: `--source vid.mp4`
* YouTube video URL: `--source 'https://youtu.be/Zgi9g1ksQHc'`
* RTSP, RTMP, or HTTP stream: `--source 'rtsp://example.com/media.mp4'`
