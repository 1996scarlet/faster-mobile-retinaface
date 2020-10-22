# Face Detection @ 500-1000 FPS

![Image of PR](docs/images/PR.webp)

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/1996scarlet/faster-mobile-retinaface.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/1996scarlet/faster-mobile-retinaface/context:python)
![License](https://badgen.net/github/license/1996scarlet/faster-mobile-retinaface)

100% Python3 reimplementation of [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace),

* Replacing cuda anchors generator with numpy api.
* [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)

## Getting Start

### For Jetson-Nano

[Install MXNet on a Jetson](https://mxnet.apache.org/get_started/jetson_setup)

* [Install gstreamer for reading videos](https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c)

    ```shell
    sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
    ```

* Testing on usb camera 0  &nbsp; (**Be Careful About `!` and `|`**)

    ```shell
    gst-launch-1.0 -q v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw, format=BGR ! fdsink | python3 face_detector.py
    ```

* For video files testing:

    ```bash
    gst-launch-1.0 -q filesrc location=$YOUR_FILE_PATH ! qtdemux ! h264parse ! avdec_h264 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw, format=BGR ! fdsink | python3 face_detector.py
    ```

## Why Should We Use Faster-RetinaFace

For middle-close range face detection,

去掉landmark分支

FPN层数削减 金字塔层数

锚框数量降低

对于图像分辨率适中且距离相对较近的应用场景, 可以适当精简锚框的种类与数量以降低运算量.

的人脸检测

## Experiments

Plan | Inference | Postprocess | Throughput Capacity (FPS)
--------|-----|--------|---------
9750HQ+1660TI | 0.9ms | 1.5ms | 500~1000
Jetson-Nano | 4.6ms | 11.4ms | 80~200

If the queue is bigger enough, the throughput capacity can reach the highest.

包括预处理, 推断, 后处理. 由于分辨率上升时, 推断速度会显著增长, 我们重点优化的预处理与后处理过程, 在整个处理过程中的占比降低, 因此加速效果降低. GTX 1660Ti with CUDA 10.2 on Platform KDE UBUNTU 20.04

VGA-Scale | RetinaFace | Faster RetinaFace | Speed Up
--------|-----|--------|---------
0.1 | 2.854ms | 2.155ms | 32%
0.4 | 3.481ms | 2.916ms | 19%
1.0 | 5.743ms | 5.413ms | 6.1%
2.0 | 22.351ms | 20.599ms | 8.5%

## Citation

``` bibtex
@inproceedings{deng2019retinaface,
    title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
    author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
    booktitle={arxiv},
    year={2019}
}
```
