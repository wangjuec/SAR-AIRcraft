import streamlit as st
import os
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
from ultralytics import YOLO

# 设置页面配置，启用宽屏模式，设置页面宽度为1000像素
st.set_page_config(layout="wide", 
                   initial_sidebar_state="expanded", 
                   page_title="YOLO 目标检测演示", 
                   page_icon=":airplane:"
                   )

# 设置文件夹路径
labels_folder = "labels"
models_folder = "models"
images_folder = "images"

# Streamlit 应用标题
st.title("光环4组：YOLO :airplane:目标检测演示 ")

# 选择示例图片
with st.sidebar:
    image_files = sorted([f for f in os.listdir(images_folder) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))
                         ])
    selected_image = st.selectbox("选择示例图片", image_files)

if selected_image:
    # 显示原始图片
    image_path = os.path.join(images_folder, selected_image)
    image = Image.open(image_path)
    
    # 加载标记
    label_path = os.path.join(labels_folder, "annotations.xml")
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()

    # 显示原始标记图片
    image_cv_original = cv2.imread(image_path)
    for annotation in root.findall('annotation'):
        filename = annotation.find('filename').text
        if filename == selected_image:
            for obj in annotation.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                cv2.rectangle(image_cv_original, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(image_cv_original, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 进行YOLO预测
    model_path = os.path.join(models_folder, "best.pt")
    model = YOLO(model_path)
    results = model(image_path)
    image_cv_yolo = cv2.imread(image_path)
    for result in results:
        boxes = result.boxes  # Boxes 对象，包含边界框输出
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = map(int, boxes.xyxy[i])
            label = result.names[int(boxes.cls[i])]
            confidence = float(boxes.conf[i])
            cv2.rectangle(image_cv_yolo, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image_cv_yolo, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示叠加对比图片
    image_cv_combined = cv2.imread(image_path)
    for annotation in root.findall('annotation'):
        filename = annotation.find('filename').text
        if filename == selected_image:
            for obj in annotation.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                cv2.rectangle(image_cv_combined, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(image_cv_combined, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    for result in results:
        boxes = result.boxes  # Boxes 对象，包含边界框输出
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = map(int, boxes.xyxy[i])
            label = result.names[int(boxes.cls[i])]
            confidence = float(boxes.conf[i])
            cv2.rectangle(image_cv_combined, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image_cv_combined, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 创建2*2网格布局
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.image(image, caption="原始图片", use_column_width=True)
    col2.image(image_cv_original, caption="原始标记图片", use_column_width=True)
    col3.image(image_cv_yolo, caption="YOLO预测图片", use_column_width=True)
    col4.image(image_cv_combined, caption="叠加对比图片", use_column_width=True)
