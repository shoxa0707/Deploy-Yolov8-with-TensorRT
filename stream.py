import streamlit as st
import cv2
import numpy as np
import subprocess
from TensorRT.models.utils import blob, det_postprocess, letterbox
from TensorRT.models.cudart_api import TRTEngine
import tempfile

classes = ['face']

Engine = TRTEngine('fp16.engine')
H, W = Engine.inp_info[0].shape[-2:]

st.title('Face detection with TensorRT')
st.header(":green[Upload your video]")

video_data = st.file_uploader("upload", ['mp4','mov', 'avi'])

temp_file_result = "result.mp4"
converted = "convert.mp4"

with open('last.txt') as f:
    last_name, last_size = f.read().split('\n')

if video_data:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(video_data.read())
        
    if video_data.name != last_name and video_data.size != last_size:
        last_name = video_data.name
        last_size = video_data.size
        with open('last.txt', 'w') as f:
            f.write(str(last_name)+'\n'+str(last_size))
        my_bar = st.progress(0, text="Detection in progress. Please wait...")

        # read it with cv2.VideoCapture(),
        # so now we can process it with OpenCV functions
        cap = cv2.VideoCapture(temp_filename)

        nums = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(temp_file_result, fourcc, fps, (frame_width, frame_height))

        if cap.isOpened()== False:
            st.write("Error opening video stream or file. Upload another video or another video extension(Support [mp4, mov, avi]).")

        num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            my_bar.progress(num / nums, text="Detection in progress. Please wait...")
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

                out.write(draw)
            else:
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()

        subprocess.call(args=f"ffmpeg -y -i {temp_file_result} -c:v libx264 {converted}", shell=True)

        video_file = open('convert.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

        # Download button
        with open("result.mp4", "rb") as file:
            btn = st.download_button(
                    label="Download video",
                    data=file,
                    file_name="result.mp4",
                    mime="video/mp4"
                )
    else:
        video_file = open('convert.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

        # Download button
        with open("result.mp4", "rb") as file:
            btn = st.download_button(
                    label="Download video",
                    data=file,
                    file_name="result.mp4",
                    mime="video/mp4"
                )
else:
    with open('last.txt', 'w') as f:
        f.write('0'+'\n'+'0')

try:
    os.remove('result.mp4')
    os.remove('convert.mp4')
except:
    pass