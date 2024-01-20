# Hands-on Unsupervised Learning Using Python

This repo contains the code for the O'Reilly Media, Inc. book "Hands-on Unsupervised Learning Using Python: How to Build Applied Machine Learning Solutions from Unlabeled Data" by Ankur A. Patel.

Official Book Website: https://www.unsupervisedlearningbook.com/thebook

Available on Amazon: https://www.amazon.com/Hands-Unsupervised-Learning-Using-Python/dp/1492035645

Available on O'Reilly Safari: https://www.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/

More on the Author: https://www.ankurapatel.io

## Release Updates

May 2021: Added support for TensorFlow 2.x, Fashion MNIST examples, and Tensorboard for Dimensionality Reduction.

## Book Description

Many industry experts consider unsupervised learning the next frontier in artificial intelligence, one that may hold the key to the holy grail in AI research, the so-called general artificial intelligence. Since the majority of the world's data is unlabeled, conventional supervised learning cannot be applied; this is where unsupervised learning comes in. Unsupervised learning can be applied to unlabeled datasets to discover meaningful patterns buried deep in the data, patterns that may be near impossible for humans to uncover.

Author Ankur Patel provides practical knowledge on how to apply unsupervised learning using two simple, production-ready Python frameworks - scikit-learn and TensorFlow. With the hands-on examples and code provided, you will identify difficult-to-find patterns in data and gain deeper business insight, detect anomalies, perform automatic feature engineering and selection, and generate synthetic datasets. All you need is programming and some machine learning experience to get started.

* Compare the strengths and weaknesses of the different machine learning approaches: supervised, unsupervised, and reinforcement learning
* Set up and manage a machine learning project end-to-end - everything from data acquisition to building a model and implementing a solution in production
* Use dimensionality reduction algorithms to uncover the most relevant information in data and build an anomaly detection system to catch credit card fraud
* Apply clustering algorithms to segment users - such as loan borrowers - into distinct and homogeneous groups
* Use autoencoders to perform automatic feature engineering and selection
* Combine supervised and unsupervised learning algorithms to develop semi-supervised solutions
* Build movie recommender systems using restricted Boltzmann machines
* Generate synthetic images using deep belief networks and generative adversarial networks
* Perform clustering on time series data such as electrocardiograms
* Explore the successes of unsupervised learning to date and its promising future

## Google Colaboratory

If you wish use Google Colab (instead of your local machine), please follow [these instructions to run the code on Google Colab](https://colab.research.google.com/github/aapatel09/handson-unsupervised-learning/blob/master/google_colab_setup.ipynb).

## Setup Main Conda Environment

If you wish to run this repo on your local machine, please follow these instructions below.

1) If you are on macOS, install Xcode Command Line Tools using ```xcode-select --install``` in Terminal.

2) Install the [Miniforge distribution of Python 3.8](https://github.com/conda-forge/miniforge#download) based on your OS. If you are on Windows, you can choose the [Anaconda distribution of Python 3.8](https://www.anaconda.com/products/individual) instead of the Miniforge distribution, if you wish to.

3) For NVIDIA GPU support, install [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive). This is only available on select NVIDIA GPUs.

4) Set up new Anaconda environment and follow these instructions based on your OS.

For **Windows**:

    ```
	conda env create -f environment_windows.yml
	conda activate unsupervisedLearning
	pip install -r requirements_windows.txt
    ```

For **macOS**:

    ```
    conda env create -f environment_mac.yml
	conda activate unsupervisedLearning
	pip install -r requirements_mac.txt
    ```

5) Download data from Google Drive (files are too large to store and access on Github).

    ```
	https://drive.google.com/drive/folders/1TQVOPUU4tVOYZvdpbxUo6uOCh0jvWNhv?usp=sharing
    ```

6) Run the notebooks using Jupyter.

    ```
	jupyter notebook
    ```

7) If you encounter any issues or errors with the setup or the code or anything else, please email the author at ankur@unsupervisedlearningbook.com.

## Set up TensorFlow for macOS Conda Environment

Please follow these instructions to set up TensorFlow for macOS.

For **macOS**:

    ```
    conda env create -f environment_tensorflow_mac.yml
	conda activate tensorflow_mac
	pip install -r requirements_tensorflow_mac.txt

	For Apple Silicon Mac (M1):
		pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl

	For Intel Mac:
		pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_x86_64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_x86_64.whl 
    ```

Please refer to this [TensorFlow for macOS guide](https://github.com/apple/tensorflow_macos/issues/153) if you run into issues or contact us.
