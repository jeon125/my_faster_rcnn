import streamlit as st
import torch
import gdown  # Google Drive에서 모델 다운로드
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
from torchvision.ops import nms

# ✅ Google Drive에서 모델 다운로드 함수
def download_model(model_name, drive_link):
    model_path = f"models/{model_name}"
    if not os.path.exists(model_path):  # 모델이 없으면 다운로드
        os.makedirs("models", exist_ok=True)  # 폴더 생성
        gdown.download(drive_link, model_path, quiet=False)
    return model_path

# ✅ Google Drive 공유 링크 설정 (ID 변경 필요)
model_links = {
    "best_model.pth": "https://drive.google.com/uc?id=11u2cUNul_DZ0bJmykxXr90c9xYQD13CA",
    "best_model_fold_1.pth": "https://drive.google.com/uc?id=13JPuwcQLSCkYfl--9zqVngf5nv-R0pSk",
    "best_model_fold_2.pth": "https://drive.google.com/uc?id=1xa0X7KbHPL5VvFd1Ue6DpYlRSXOYeU5X",
    "best_model_fold_3.pth": "https://drive.google.com/uc?id=1j23rzkwYMmUaCGeh0rEJco2W3Zt3VDsE",
    "best_model_fold_4.pth": "https://drive.google.com/uc?id=1Vdu2FsHYW7evr68htF4hp6wTu6BIc9jy",
    "best_model_fold_5.pth": "https://drive.google.com/uc?id=1VPHxGtXz5SSPFQ11xpws2Ng7mbjANeth",
}

# ✅ Faster R-CNN 모델 로드 함수
@st.cache_resource
def load_model(model_path, num_classes, device):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ✅ 모델 및 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 다운로드 및 로드
model_path_single = download_model("best_model.pth", model_links["best_model.pth"])  # 단일 모델
model_paths_kfold = [
    download_model(f"best_model_fold_{i}.pth", model_links[f"best_model_fold_{i}.pth"])
    for i in range(1, 6)
]  # K-Fold 모델들

# ✅ 클래스 인덱스 → 이름 변환 딕셔너리
labels_inv = {1: "extruded", 2: "crack", 3: "cutting", 4: "side_stamped"}

# ✅ Streamlit GUI 시작
st.title("🔍 O-ring 불량 확인")
st.markdown("### : 훈련된 Faster R-CNN 모델을 사용하여 결함 탐지")

# ✅ 모델 선택
st.sidebar.markdown("<h3 style='font-size:20px;'>🛠 사용할 모델을 선택하세요</h3>", unsafe_allow_html=True)
model_option = st.sidebar.selectbox("", ["단일 모델", "K-Fold 앙상블"])

# ✅ 모델 로드
if model_option == "K-Fold 앙상블":
    models = [load_model(path, num_classes=5, device=device) for path in model_paths_kfold]
else:
    model = load_model(model_path_single, num_classes=5, device=device)

# ✅ 이미지 업로드
st.sidebar.markdown("<h3 style='font-size:20px;'>📂 이미지를 업로드하세요</h3>", unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# ✅ 이미지 인덱스 저장
if "image_index" not in st.session_state:
    st.session_state.image_index = 0

if uploaded_files:
    num_files = len(uploaded_files)

    # ✅ 슬라이더 추가
    st.markdown("<h3 style='font-size:20px;'>📷 선택된 이미지</h3>", unsafe_allow_html=True)
    st.session_state.image_index = st.slider("", min_value=0, max_value=num_files-1, value=st.session_state.image_index)
    
    # ✅ 현재 이미지 선택
    uploaded_file = uploaded_files[st.session_state.image_index]

    # ✅ 이미지 로드 및 전처리
    image = Image.open(uploaded_file).convert("RGB")
    
    # ✅ OpenCV 변환 후 전처리
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        margin = 20
        x_new, y_new = max(0, x - margin), max(0, y - margin)
        x_end, y_end = min(img_cv.shape[1], x + w + margin), min(img_cv.shape[0], y + h + margin)
        cropped = img_cv[y_new:y_end, x_new:x_end]
        resized_image = cv2.resize(cropped, (500, 500))
        image = Image.fromarray(resized_image)

    # ✅ 이미지 텐서 변환
    img_tensor = F.to_tensor(image).to(device)

    # ✅ 모델 예측 수행
    if model_option == "단일 모델":
        with torch.no_grad():
            pred = model([img_tensor])[0]
        final_boxes, final_scores, final_labels = pred["boxes"], pred["scores"], pred["labels"]
    else:
        all_preds = [model([img_tensor])[0] for model in models]
        img_boxes = torch.cat([pred["boxes"] for pred in all_preds], dim=0)
        img_scores = torch.cat([pred["scores"] for pred in all_preds], dim=0)
        img_labels = torch.cat([pred["labels"] for pred in all_preds], dim=0)

        iou_threshold, score_threshold = 0.3, 0.5
        keep = img_scores > score_threshold
        img_boxes, img_scores, img_labels = img_boxes[keep], img_scores[keep], img_labels[keep]
        keep_indices = nms(img_boxes, img_scores, iou_threshold)
        final_boxes, final_scores, final_labels = img_boxes[keep_indices], img_scores[keep_indices], img_labels[keep_indices]

    # ✅ 결과 표시
    result_text = "⚠ 결함 존재!" if len(final_boxes) > 0 else "✅ 결함 없음!"
    text_color = "red" if len(final_boxes) > 0 else "black"
    st.markdown(f"<h2 style='text-align: center; color: {text_color};'>{result_text}</h2>", unsafe_allow_html=True)

    img_with_boxes = draw_bounding_boxes((img_tensor * 255).byte(), final_boxes, colors="red", width=3)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_with_boxes.permute(1, 2, 0).cpu())
    ax.axis("off")
    st.pyplot(fig)
