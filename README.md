# Deploy Yolov8 model on TensorRT

## Introduction

NVIDIA [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html) is an SDK for optimizing trained deep learning models to enable high-performance inference. TensorRT contains a deep learning inference optimizer for trained deep learning models, and a runtime for execution.

<img src="images/tensorrt.png">

After you have trained your deep learning model in a framework of your choice, TensorRT enables you to run it with higher throughput and lower latency.<br>
You need a device with a GPU to use this project.

## Convert model

First we convert trained model(with .pt extension) to TensorRT (.engine extension). For this we need to switch from pytorch model to onnx first:

```console
  Usage: python pt2onnx.py -m model.pt -s model.onnx [options]...

  A common command: python pt2onnx.py -m some_model.pt

    -m --model               Pytorch model path(.pt). Default: model.pt
    -s --save                Onnx model save path(.onnx). Default: model.onnx
```

Then we have an onx model. Through it we can go to TensorRT. This is done as follows:

```console
  Usage: python onnx2engine.py -m model.onnx -p fp16 -s model.engine [options]...

  A common command: python onnx2engine.py -m some_model.onnx

    -m --model               Pytorch model path(.pt). Default: model.pt
    -p --precision           Model quantization. Options: fp16 | fp32 | int8. Default: fp16
    -s --save                TensorRT model save path(.engine). Default: model.engine
```

## ⚡ Quick Inference

### Python script

If you want, you can use the TensorRT model with python code. You can do this as follows:

```console
  Usage: python deploy.py -t tensorflow -m models/seatbelt.model -i some.png -d cpu [options]...

  A common command: python inference.py -i some.png

    -m --model               Pytorch model path(.pt). Default: fp16.engine
    -v --video               Video path. Default video.mp4
    -c --classes             Text file where classes names are stored. Default: classes.txt
    -o --output              Output type. Options: opencv - programm will out frames with opencv(cv2) | write - result will store in file(write .mp4 file). Default: opencv
    -s --save                Stored video path. If "output" type is "write", save results to video. Default: video.mp4
```

## 👀 Demos

Used streamlit for deploy. [Streamlit](https://github.com/streamlit/streamlit) lets you turn data scripts into shareable web apps in minutes, not weeks. It’s all Python, open-source, and free! And once you’ve created an app you can use our Community Cloud platform to deploy, manage, and share your app.

You can run following command to use streamlit in our case:

```bash
  streamlit run stream.py
```

The result is as follows:

<img src="images/TensorRTDeploy.gif">

## 🔧 Dependencies and Installation

- Python >= 3.8
- PyTorch >= 2
- TensorRT >= 8

  ### Installation

  1. Clone repo

     ```bash
     git clone https://github.com/shoxa0707/SeatBelt-Classification.git
     cd SeatBelt-Classification
     ```

  1. Install dependent packages

     ```bash
     pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118(for CUDA 11.8)
     pip install -r requirements.txt
     ```

# Requirements

- Linux
- Python 3.8
- NVIDIA GPU
