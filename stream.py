import cv2
import numpy as np
import time
from TensorRT.models.utils import blob, det_postprocess, letterbox
from TensorRT.models.cudart_api import TRTEngine
import streamlit as st
import tempfile
import subprocess

classes = ['face']

Engine = TRTEngine('fp16.engine')
H, W = Engine.inp_info[0].shape[-2:]

st.title('Face detection with TensorRT')
st.header(":green[Upload your video]")

video_data = st.file_uploader("upload", ['mp4','mov', 'avi'])

write = st.checkbox('Write to video', value=True)
real = st.checkbox('Show real time')

if st.button("Let's started") and (write or real):
    if video_data:
        print(video_data)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(video_data.read())
        
        my_bar = st.progress(0, text="Detection in progress. Please wait...")

        # read it with cv2.VideoCapture(),
        # so now we can process it with OpenCV functions
        cap = cv2.VideoCapture(temp_filename)

        nums = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if write:
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('result.mp4', fourcc, fps, (frame_width, frame_height))
        if real:
            end = cv2.imread('images/end.png')
            imagepl = st.empty()

        if cap.isOpened()== False:
            st.write("Error opening video stream or file. Upload another video or another video extension(Support [mp4, mov, avi]).")

        num = 0
        cap = cv2.VideoCapture(temp_filename)

        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        with open('coco.names') as f:
            classes = f.read().split('\n')

        while(cap.isOpened()):
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
        if write:
            out.release()
            subprocess.call(args=f"ffmpeg -y -i result.mp4 -c:v libx264 convert.mp4", shell=True)

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
        if write:
            # Download button
            with open("result.mp4", "rb") as file:
                btn = st.download_button(
                        label="Download video",
                        data=file,
                        file_name="result.mp4",
                        mime="video/mp4"
                    )
        if real:
            end = cv2.imread('images/end.png')
            st.image(end)
    
