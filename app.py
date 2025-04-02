import streamlit as st
import os
from PIL import Image
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import numpy as np

try:
    import cv2
    st.success("OpenCV imported successfully!")
except Exception as e:
    st.error(f"Failed to import OpenCV: {str(e)}")

# 设置页面配置
st.set_page_config(layout="centered", 
                   initial_sidebar_state="expanded", 
                   page_title="YOLO 目标检测演示", 
                   page_icon=":airplane:"
                   )

# 设置文件夹路径
labels_folder = "labels"
models_folder = "models"
images_folder = "images"

# Streamlit 应用标题
st.title("光环51期4组：YOLO :airplane:目标检测演示 ")

# 选择示例图片或上传图片
with st.sidebar:
    image_files = sorted([f for f in os.listdir(images_folder) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))
                         ])
    selected_image = st.selectbox("选择示例图片", image_files)
    uploaded_image = st.file_uploader("或上传图片", type=["png", "jpg", "jpeg"])
    st.write("雷达图较大，请耐心等待加载！")

def load_image(image_path):
    return Image.open(image_path)

@st.cache_resource
def load_annotations(label_path):
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        return tree.getroot()
    return None

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def draw_annotations(image, annotations, color, label_color):
    image_name = uploaded_image.name if uploaded_image else selected_image
    for annotation in annotations:
        filename = annotation.find('filename').text
        if filename == image_name:
            for obj in annotation.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
    return image

def draw_yolo_predictions(image, results, color, label_color):
    for result in results:
        boxes = result.boxes  # Boxes 对象，包含边界框输出
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = map(int, boxes.xyxy[i])
            label = result.names[int(boxes.cls[i])]
            confidence = float(boxes.conf[i])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
    return image

if uploaded_image:
    image = Image.open(uploaded_image)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
else:
    image_path = os.path.join(images_folder, selected_image)
    image = load_image(image_path)
    image_cv = cv2.imread(image_path)

# 加载标记
label_path = os.path.join(labels_folder, "annotations.xml")
root = load_annotations(label_path)

# 显示原始标记图片
image_cv_original = image_cv.copy()
image_cv_original = draw_annotations(image_cv_original, root.findall('annotation'), (255, 0, 0), (255, 0, 0))

# 进行YOLO预测
model_path = os.path.join(models_folder, "best.pt")
model = load_model(model_path)
results = model(image_cv)
image_cv_yolo = image_cv.copy()
image_cv_yolo = draw_yolo_predictions(image_cv_yolo, results, (0, 255, 0), (0, 255, 0))

# 显示叠加对比图片
image_cv_combined = image_cv.copy()
image_cv_combined = draw_annotations(image_cv_combined, root.findall('annotation'), (255, 0, 0), (255, 0, 0))
image_cv_combined = draw_yolo_predictions(image_cv_combined, results, (0, 255, 0), (0, 255, 0))

# 创建2*2网格布局
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.image(image, caption="原始图片")
col2.image(image_cv_original, caption="原始标记图片")
col3.image(image_cv_yolo, caption="YOLO预测图片")
col4.image(image_cv_combined, caption="叠加对比图片")
