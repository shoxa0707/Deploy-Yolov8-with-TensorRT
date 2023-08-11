import cv2
import numpy as np
from TensorRT.models.utils import blob, det_postprocess, letterbox
from TensorRT.models.cudart_api import TRTEngine
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model path", type=str, default="fp16.engine")
parser.add_argument("-v", "--video", help="video path", type=str, default="video.mp4")
parser.add_argument("-c", "--classes", help="text file where classes are stored", type=str, default="classes.txt")
parser.add_argument("-o", "--output", help="output type", type=str, default="opencv")
parser.add_argument("-s", "--save", help="save path", type=str, default="output.mp4")
args = parser.parse_args()

assert args.output in ["opencv", "write"], "output argument support only following values:\n    opencv - programm will out frames with opencv(cv2)\n    write - result will store in file(write .mp4 file)"

with open(args.classes) as f:
    classes = f.read().split("\n")

Engine = TRTEngine(args.model)
H, W = Engine.inp_info[0].shape[-2:]

cap = cv2.VideoCapture(args.video)

nums = cap.get(cv2.CAP_PROP_FRAME_COUNT)

if args.output == "write":
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.save, fourcc, fps, (frame_width, frame_height))

if (cap.isOpened()== False):
    print("Error opening video stream or file")
print("Process has started...")

num = 1
while cap.isOpened():
    ret, frame = cap.read()
    if num < nums - 1:
        about = f"""[{int((num+1)/nums*30)*"="}>{(29-int((num+1)/nums*30))*"."}] - Frame: {num}"""
        print(about, end='\r')
    elif num == nums - 1:
        about = f"""[{30*"="}] - Frame: {num}"""
        print(about)
    num += 1
    if ret == True:
        draw = frame.copy()
        bgr, ratio, dwdh = letterbox(frame, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        # inference
        data = Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().astype(np.int32).tolist()
            cls = classes[int(label)]
            color = (255, 0, 0)
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw, f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)

        if args.output == "opencv":
            cv2.imshow('output', draw)
            # Press q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            out.write(draw)
    else:
        break
print("Process ended.")
if args.output == "write":
    print(f"File saved in {args.save}")
    out.release()
cap.release()
cv2.destroyAllWindows()
