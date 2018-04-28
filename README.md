# Team Vav Project Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The task for the Kaggle DataScience Bowl, 2018 is to create an algorithm to automate nucleus detection. This project has been our teams attempt to come up with a solution for this problem.

More details can be found [here](https://www.kaggle.com/c/data-science-bowl-2018#description).

## Getting Started

If you follow the below instructions it will allow you to install and run the training or testing.

### Prerequisites

What things you need to install the software and how to install them

- [Python 2.7](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/) - Python Environment virtualization. 
- [Keras](https://keras.io/) Deep Learning API for Tensorflow.
- [Tensorflow](https://www.tensorflow.org/) Deep Learning Library.

### Installing

#### Anaconda

Anaconda is a complete Python distribution embarking automatically the most common packages, and allowing an easy installation of new packages.

Download and install Anaconda from (https://www.continuum.io/downloads).
The link for Linux,Mac and Windows are in the website.Following their instruction will install the tool.
##### Running Environment

* Once Anaconda is installed open anaconda prompt(Windows/PC) Command Line shell(Mac OSX or Unix)
* Run ```conda env create -f environment.yml``` will install all packages required for all programs in this repository
###### To start the environment 

* For Unix like systems ```source activate vav```

* For PC like systems ```activate vav```

#### Keras

You can install keras using ``` pip ``` on command line
``` sudo pip install keras ```

The `environment.yml` file for conda is placed in [Extra](https://github.com/dsp-uga/vav/blob/master/environment.yml) for your ease of installation this has keras

#### Tensorflow
Installing Tensorflow is straight forward using ``` pip ``` on command line

* If CPU then  ``` sudo pip install tensorflow ```
* If GPU then ``` sudo pip install tensorflow-gpu ```


The `environment.yml` file for conda is placed in [Extra](https://github.com/dsp-uga/vav/blob/master/environment.yml) for your ease of installation this has tensorflow.

#### Downloading the dataset (Optional)

Refer to downloading the dataset on this page: [Data](https://www.kaggle.com/c/data-science-bowl-2018/data)

## Data

This dataset contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). The dataset is designed to challenge an algorithm's ability to generalize across these variations. For more description on the dataset please refer [here] (https://www.kaggle.com/c/data-science-bowl-2018/data).

## Running and Training

One can run `unet.py` via regular **python** 

```
$ python unet.py [train or Test] [optional args]
```
Example: ```python unet.py train ```

  - **Required Arguments**

    - `trainortest`: This is a string either train or test

  - **Optional Arguments**

    - `-batch-size`: The batch size if applicable (Default: `20`)
    - `-masks`: Path to the masks directory where masks are present. (Default: `train\masks`)
    - `-dataset`: Path to the dataset directory where train dataset is present. (Default: `train\`)


## Results

Method| Mean IoU Score (on Kaggle Board)
--- |  ---
Threshold UNet [1]  | 0.278
Thresholding + Median Filter UNet [4] |0.335

## Authors

* **Ankita Joshi** - [AnkitaJo](https://github.com/AnkitaJo)
* **Vibhod Phenani** - [vibodh01](https://github.com/vibodh01)
* **Vyom Srivastava** - [vyom1911](https://github.com/vyom1911)

See also the list of [contributors](https://github.com/dsp-uga/vav/blob/master/Contributors.md) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/dsp-uga/vav/blob/master/LICENSE) file for details

## Acknowledgments and References

* Hat tip to anyone who's code was used


