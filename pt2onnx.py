from io import BytesIO
import onnxsim
import torch
from TensorRT.models.common import PostDetect, optim
from ultralytics import YOLO
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model path", type=str, default="face.pt")
parser.add_argument("-s", "--save", help="model save path", type=str, default="face.onnx")
args = parser.parse_args()

PostDetect.conf_thres = 0.25
PostDetect.iou_thres = 0.65
PostDetect.topk = 100

b = 1
YOLOv8 = YOLO(args.model)
model = YOLOv8.model.fuse().eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for m in model.modules():
    optim(m)
    m.to(device)
model.to(device)
fake_input = torch.randn((1,3,640,640)).to(device)
for _ in range(2):
    model(fake_input)

with BytesIO() as f:
    torch.onnx.export(
        model,
        fake_input,
        f,
        opset_version=11,
        input_names=['images'],
        output_names=['num_dets', 'bboxes', 'scores', 'labels'])
    f.seek(0)
    onnx_model = onnx.load(f)
onnx.checker.check_model(onnx_model)
shapes = [b, 1, b, 100, 4, b, 100, b, 100]
for i in onnx_model.graph.output:
    for j in i.type.tensor_type.shape.dim:
        j.dim_param = str(shapes.pop(0))
try:
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, 'assert check failed'
except Exception as e:
    print(f'Simplifier failure: {e}')
onnx.save(onnx_model, args.save)
print(f'ONNX export success, saved as {args.save}')
