import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from base64 import b64encode
import tempfile
import math

focal_length = 0
distance_to_object_m = 0
width = 0
height = 0


def save_uploaded_file(uploaded_file):   
    with Image.open(uploaded_file) as img:
        width, height = img.size 
    # st.success(f"image width {width} ")
    # st.success(f"image height {height} ")
    
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


file_path = ''


# def calculate_object_real_width(image, det):

#     sensor_size = model_sensor_size[mobile_model_name]
#     sensor_width_mm = sensor_size["sensor_width_mm"]
#     sensor_height_mm = sensor_size["sensor_height_mm"]
#     sensor_width_px = sensor_size["sensor_width_px"]
#     sensor_height_px = sensor_size["sensor_height_px"]

#     # st.success(f"mobile_model_name {mobile_model_name} ")
#     # st.success(f"sensor_width_mm {str(sensor_width_mm)} ")
#     # st.success(f"sensor_height_mm {str(sensor_height_mm)} ")
#     # st.success(f"sensor_width_px {str(sensor_width_px)} ")
#     # st.success(f"sensor_height_px {str(sensor_height_px)} ")

#     image_width = image_heigth = 0

#     with Image.open(uploaded_file) as img:
#         width, height = img.size


#     x1, y1, x2, y2 = map(int, det[:4])

#     # st.success(f"x1 {str(x1)} ")
#     # st.success(f"x2 {str(x2)} ")
#     # st.success(f"y1 {str(y1)} ")
#     # st.success(f"y2 {str(y2)} ")

#     if width > height :
#         image_width = width
#         image_heigth = height
#         Object_width_pixels = x2 - x1
#         Object_height_pixels = y2 - y1
#     else:
#         image_width = height
#         image_heigth = width
#         Object_width_pixels = y2 - y1
#         Object_height_pixels = x2 - x1

#     # st.success(f"Object_width_pixels {str(Object_width_pixels)} ")
#     # st.success(f"Object_height_pixels {str(Object_height_pixels)} ")

#     # st.success(f"image_width {str(image_width)} ")
#     # st.success(f"image_heigth {str(image_heigth)} ")

#     distance_to_object_mm = int(distance_to_object_m) * 1000

#     # st.success(f"distance_to_object_mm {str(distance_to_object_mm)} ")

#     Object_width_on_sensor_mms = (sensor_width_mm * Object_width_pixels) / sensor_width_px
#     Object_width_mm =((distance_to_object_mm * Object_width_on_sensor_mms)/float(focal_length))

#     Object_height_on_sensor_mms = (sensor_height_mm * Object_height_pixels) / sensor_height_px
#     Object_height_mm = ((distance_to_object_mm * Object_height_on_sensor_mms) / float(focal_length))

#     # st.success(f"Object_width_on_sensor_mms {str(Object_width_on_sensor_mms)} ")
#     # st.success(f"Object_height_on_sensor_mms {str(Object_height_on_sensor_mms)} ")

#     # st.success(f"Object_width_mm {str(Object_width_mm)} ")
#     # st.success(f"Object_height_mm {str(Object_height_mm)} ")

#     return Object_width_mm, Object_height_mm

def calculate_object_real_width(image, det, sensor_size):
    
    sensor_width_mm = sensor_size['sensor_width_mm']
    sensor_height_mm = sensor_size['sensor_height_mm']
    sensor_width_px = sensor_size['sensor_width_px']
    sensor_height_px = sensor_size['sensor_height_px']
    # focal_len = focal_length
    st.success(f"focal_length-2 {focal_length} ")
    
    image_width = image_heigth = 0
    width, height = image.size
    x1, y1, x2, y2 = map(int, det[:4])    

    image_width, image_height = (width, height) if width > height else (height, width)
    Object_width_pixels, Object_height_pixels = (x2 - x1, y2 - y1) if width > height else (y2 - y1, x2 - x1)

    distance_to_object_mm = int(distance_to_object_m) * 1000 # Distance conversion
    
    # Calculate dimensions on sensor
    
    Object_width_on_sensor_mms = (sensor_width_mm * Object_width_pixels) / sensor_width_px
    Object_width_mm =((distance_to_object_mm * Object_width_on_sensor_mms)/float(focal_length))

    # Calculate real dimensions
    Object_height_on_sensor_mms = (sensor_height_mm * Object_height_pixels) / sensor_height_px
    Object_height_mm = ((distance_to_object_mm * Object_height_on_sensor_mms) / float(focal_length))
    
    return Object_width_mm, Object_height_mm

def draw_detections(image, detections,selected_sensor_data):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(30)
    
    for det in detections:
        # st.success(f"focal_length {str(focal_length)} ")
        
        real_world_width_mm, real_world_length_mm = calculate_object_real_width(image, det,selected_sensor_data)
        real_world_length_cm = int(real_world_length_mm / 10)
        real_world_width_cm = int(real_world_width_mm / 10)

        # st.success(f"real_world_width_cm {str(real_world_width_cm)} ")
        # st.success(f"real_world_length_cm {str(real_world_length_cm)} ")

        area_cm2 = real_world_length_cm * real_world_width_cm

        x1, y1, x2, y2 = map(int, det[:4])
        # label = f'{int(cls_id)} {conf:.2f} | Area: {area_m2:.2f} m²'
        label = f'Area: {area_cm2} Sq Cm'
        # st.success(f"Area    --    {area_cm2} Sq Cm ")
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)
        text_width = font.getlength(label)
        text_height = font.getbbox(label)[3]  # Get the height from the bounding box
        draw.rectangle([x1, y1, x1 + text_width, y1 + text_height + 2], fill=(255, 0, 0), outline=(255, 0, 0))
        draw.text((x1, y1), label, font=font, fill=(255, 255, 255))
    
    return image

def generate_roadmap(image, detections, threshold=5):
    road_blocked = len(detections) > threshold
    if road_blocked:
        draw = ImageDraw.Draw(image)
        draw.line([(0, 0), (image.width, image.height)], fill=(255, 0, 0), width=10)
        #draw.text((10, 10), "Road Blocked", font=ImageFont.truetype("arial.ttf", 36), fill=(255, 0, 0))
    return image, road_blocked

st.title('Pothole Detection')
current_dir = os.path.dirname(__file__)
# Model selection
model_version = st.selectbox('Choose your model:', ('yolo-v1', 'yolo-v2'))
model_paths = {
    'yolo-v1': current_dir + '//content//runs//detect//train5//best.pt',
    'yolo-v2': current_dir + '//content//runs//detect//train5//weights//best.pt'
}
model_path = model_paths[model_version]
model = YOLO(model_path)

mobile_model_name = st.selectbox('Choose your mobile:', ('Iphone 14', 'Pixel 6A', 'One Plus','motorola edge 40 neo', 'iQOO Neo6'))
model_sensor_size =  {
        "Iphone 14":{ "sensor_width_mm":7, "sensor_height_mm":5, "sensor_width_px":4032, "sensor_height_px":3024, "focal_length":7.5,"Distance_to_object":2},
        "iQOO Neo6":{ "sensor_width_mm":7.4, "sensor_height_mm":5.5, "sensor_width_px":9280, "sensor_height_px":6944, "focal_length":5,"Distance_to_object":2},
        "Pixel 6A":{ "sensor_width_mm":7.68, "sensor_height_mm":5.76, "sensor_width_px":4032, "sensor_height_px":3024, "focal_length":4.38,"Distance_to_object":2},
        "One Plus":{ "sensor_width_mm":7.4, "sensor_height_mm":5.5, "sensor_width_px":4032, "sensor_height_px":3024, "focal_length":5.59,"Distance_to_object":2},
        "motorola edge 40 neo":{"sensor_width_mm":8, "sensor_height_mm":6, "sensor_width_px":8160, "sensor_height_px":6120, "focal_length":6,"Distance_to_object":2}
    }
selected_sensor_data = model_sensor_size.get(mobile_model_name, {})

focal_length = st.text_input(
    label="Focal Length in mm",
    value=str(selected_sensor_data.get("focal_length", "")),
    max_chars=6,
    key="focal_length",
    type="default",
    autocomplete="off")

distance_to_object_m = st.text_input(
    label="Distance in meters",
    value=str(selected_sensor_data.get("Distance_to_object", "")), # units in m
    max_chars=6,
    key="distance_to_object",
    type="default",
    autocomplete="off")

uploaded_file = st.file_uploader("Upload an image or video...", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    
    file_path = save_uploaded_file(uploaded_file)

    if uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):

        results = model(file_path, conf=0.1)

        # Iterate over the results and display each processed image
        for result in results:
            result_image = result.orig_img
            # image_pil = Image.fromarray(result_image)
            # # Get the width and height
            # image_width, image_height = image_pil.size
            # # width, height = result_image
            # st.success(f"height {image_width} , {image_height} ")
            st.image(result_image, caption='Processed Image', use_column_width=True)

            detections = result.boxes.data.tolist()
            if detections:
                annotated_image = Image.fromarray(result_image)
                annotated_image = draw_detections(annotated_image, detections, selected_sensor_data)
                roadmap_image, road_blocked = generate_roadmap(annotated_image, detections, threshold=5)
                st.image(roadmap_image, caption='Roadmap', use_column_width=True)
                if road_blocked:
                    st.warning("Too many potholes detected. Road is blocked for cars.")
                else:
                    st.success("Road is clear for vehicles to drive through.")
            else:
                st.success("No Detections identified for the above media ")

    elif uploaded_file.name.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            st.error("Failed to open the video file.")
        else:
            st.write("Processing video...")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)

            # Create a temporary directory for storing processed frames
            with tempfile.TemporaryDirectory() as temp_dir:
                processed_frames = []

                for frame_idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=0.1)

                    for result in results:
                        detections = result.boxes.data.tolist()
                        if detections:
                            result_image = result.orig_img
                            annotated_image = Image.fromarray(result_image)
                            annotated_image = draw_detections(annotated_image, detections)
                            roadmap_image, road_blocked = generate_roadmap(annotated_image, detections, threshold=5)
                            st.image(roadmap_image, caption='Processed Frame', use_column_width=True)
                            if road_blocked:
                                st.warning("Too many potholes detected. Road is blocked for cars.")

                            # Save the processed frame to the temporary directory
                            frame_path = os.path.join(temp_dir, f"frame_{frame_idx}.jpg")
                            roadmap_image.save(frame_path)
                            processed_frames.append(frame_path)

                    progress_bar.progress((frame_idx + 1) / frame_count)

                cap.release()

                # Combine the processed frames into a video
                output_video_path = os.path.join(temp_dir, "output_video.mp4")
                frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (int(frame_width), int(frame_height)))

                for frame_path in processed_frames:
                    frame = cv2.imread(frame_path)
                    video_writer.write(frame)

                video_writer.release()

                # Display the combined video
                video_bytes = open(output_video_path, "rb").read()
                st.video(video_bytes)

    # Cleanup: Remove the uploaded file to clear space
    os.remove(file_path)
