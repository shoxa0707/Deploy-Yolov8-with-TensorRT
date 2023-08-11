from TensorRT.models import EngineBuilder
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model path", type=str, default="face.onnx")
parser.add_argument("-p", "--precision", help="Precision. Model quantzation", type=str, default="fp16")
parser.add_argument("-s", "--save", help="Model save path", type=str, default="fp16.engine")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

builder = EngineBuilder(args.model, device)
builder.seg = True
builder.weight = args.save
builder.build(precision=args.precision,
              input_shape=[1, 3, 640, 640],
              iou_thres=0.65,
              conf_thres=0.25,
              topk=100)
