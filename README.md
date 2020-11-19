# Deep Multi-view Depth Estimation with Predicted Uncertainty

# Abstract

In this paper, we address the problem of estimating dense depth from a sequence of images using deep neural networks. Specifically, we employ a dense-optical-flow network to compute correspondences and then triangulate the point cloud to obtain an initial depth map. Parts of the point cloud, however, may be less accurate than others due to lack of common observations or small baseline-to-depth ratio. To further increase the triangulation accuracy, we introduce a depth-refinement network (DRN) that optimizes the initial depth map based on the image’s contextual cues. In particular, the DRN contains an iterative refinement module (IRM) that improves the depth accuracy over iterations by refining the deep features. Lastly, the DRN also predicts the uncertainty in the refined depths, which is desirable in applications such as measurement selection for scene reconstruction. We show experimentally that our algorithm outperforms state-of-the-art approaches in terms of depth accuracy, and verify that our predicted uncertainty is highly correlated to the actual depth error.

# Setting up Environment

For convenience, the code are assumed to be run inside NVIDIA-Docker. For instructions on installing NVIDIA-Docker, please follow the following steps (note that this is for Ubuntu 18.04):

For more detailed instructions, please refer to [this link](https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/).

1. Install Docker

    ```
    sudo apt-get update

    sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent \
        software-properties-common

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

    sudo apt-key fingerprint 0EBFCD88

    sudo add-apt-repository \
       "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"

    sudo apt-get update

    sudo apt-get install docker-ce docker-ce-cli containerd.io
    ```

    To verify Docker installation, run:

    ```
    sudo docker run hello-world
    ```

2. Install NVIDIA-Docker

    ```
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
      sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update

    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
      sudo apt-key add -

    sudo apt-get install nvidia-docker2

    sudo pkill -SIGHUP dockerd
    ```

To activate the docker environment, run the following command:

```
nvidia-docker run -it --rm --ipc=host -v /:/home nvcr.io/nvidia/pytorch:20.03-py3
```

where `/` is the directory in the local machine (in this case, the root folder), and `/home` is the reflection of that directory in the docker.
This has also specified NVIDIA-Docker with PyTorch version 20.03 which is required to ensure the compatibility
between the packages used in the code (at the time of submission).

Inside the docker, change the working directory to this repository:
```
cd /home/PATH/TO/THIS/REPO/DeepMultiviewDepth
```

# Run Demo
1. Please download and extract the files provided by this [link](https://drive.google.com/file/d/1mNiZaDtmzxNSjHGUAZy9qwQ8Ek3ffJJO/view?usp=sharing) to `./checkpoints/` directory.
