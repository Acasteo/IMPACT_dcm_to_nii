# IMPACT DICOM To Nifti Conversion

This is a Python-based pipeline for converting medical images from DICOM format to nifti or mhd format. The pipeline is designed to simplify the process of preprocessing medical imaging data for AI model development.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Label JSON](#label-json)
  - [Running the Conversion](#running-the-conversion)

## Overview

Medical image data often comes in the DICOM format, which is widely used in healthcare settings. However, for many applications, it is beneficial to convert these images into the nifti or mhd format, which is more versatile and compatible with various analysis tools. 

This pipeline addresses these requirements by providing a straightforward way to:

- Convert DICOM images to nifti or mhd format.

## Features

- Converts DICOM images to nifti or mhd format.
- Supports batch processing of DICOM directories.

## Getting Started

### Prerequisites

Before you begin, make sure you have the following prerequisites installed:

- Python 3.x
- [PyDicom](https://pydicom.github.io/pydicom/stable/index.html) (for reading DICOM files)
- [SimpleITK](https://pypi.org/project/SimpleITK/) (for working with NIfTI files)
- [tqdm](https://github.com/tqdm/tqdm) (for the progress bar)
- [dcmrtstruct2nii](https://github.com/Sikerdebaard/dcmrtstruct2nii) (for the dicom conversion)
- [numpy](https://numpy.org/)

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.mdanderson.org/Morfeus-Lab/resample_pipeline.git
   ```

2. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Label JSON

The label JSON maps contour names from an RTSTRUCT to intergers for the converted mask. This allows combination of multiple contour names into 1 mask if the names are not consistent across scans. Overlapping contours are assigned the highest integar label during conversion.

Example configuration('labels_template.json')

```json
{
  "labels": {
    "label_1":1,
    "label_2":2,
    "label_3":1
  }
}
```

### Running the Conversion

To run the pipeline, use the following command:

```bash
usage: convert.py [-h] [--input INPUT] [--output OUTPUT] [--nifti] [--mhd] [-l LABEL_JSON]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to the directory containing DICOMs
  --output OUTPUT       Path to the output directory
  --nifti               Flag to convert to nii.gz volumes
  --mhd                 Flag to convert to .mhd volumes
  -l LABEL_JSON, --label_json LABEL_JSON
                        json of labels to convert
  --anon                 Flag to convert to anonymize folder
  --mist                 Flag to create mist test_path.csv
```

The pipeline will convert the DICOM files to nifti or mhd format. The resulting files will be saved in the output directory in subdirectories based on MRN.
