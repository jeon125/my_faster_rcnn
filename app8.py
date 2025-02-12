import streamlit as st
import torch
import gdown
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
import json
import io
import base64
import zipfile

# ✅ Google Drive에서 모델 다운로드 함수
def download_model(model_name, drive_link):
    model_path = f"models/{model_name}"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        gdown.download(drive_link, model_path, quiet=False)
    return model_path

# ✅ Google Drive 공유 링크 설정
model_links = {
    "best_model.pth": "https://drive.google.com/uc?id=11u2cUNul_DZ0bJmykxXr90c9xYQD13CA",
    "best_model_fold_1.pth": "https://drive.google.com/uc?id=1ta9lx56Y74ypc87f-vx7fMlofdCS3O6V",
    "best_model_fold_2.pth": "https://drive.google.com/uc?id=1KzJLz-pMGbUjIVCLBH6WS_ezAsU-nLIi",
    "best_model_fold_3.pth": "https://drive.google.com/uc?id=1eN69P2RvHW7OrdJAHL6qgAML5_ArWXW3",
    "best_model_fold_4.pth": "https://drive.google.com/uc?id=10UNC7LV06GZlqnIvmG9S9F0KUEjLRQfW",
    "best_model_fold_5.pth": "https://drive.google.com/uc?id=11wGXOe6yZZgo8wDmzHrtFiJROn3JusKl",
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
model_path_single = download_model("best_model.pth", model_links["best_model.pth"])
model_paths_kfold = [
    download_model(f"best_model_fold_{i}.pth", model_links[f"best_model_fold_{i}.pth"])
    for i in range(1, 6)
]

# ✅ 클래스 인덱스 → 이름 변환
labels_inv = {1: "extruded", 2: "crack", 3: "cutting", 4: "side_stamped"}

# ✅ Streamlit GUI 시작
st.title("🔍 O-ring 불량 확인")
st.markdown("### : 훈련된 Faster R-CNN 모델을 사용하여 결함 탐지")

# ✅ 모델 선택
st.sidebar.markdown("<h3 style='font-size:20px;'>🛠 사용할 모델을 선택하세요</h3>", unsafe_allow_html=True)
model_option = st.sidebar.selectbox("", ["단일 모델", "K-Fold 앙상블"])

# ✅ 모델 로드
if model_option == "단일 모델":
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
    if num_files > 1:
        st.session_state.image_index = st.slider("", min_value=0, max_value=num_files-1, value=st.session_state.image_index)
    else:
        st.session_state.image_index = 0

    # ✅ 현재 이미지 선택
    uploaded_file = uploaded_files[st.session_state.image_index]
    image_name = uploaded_file.name
    image = Image.open(uploaded_file).convert("RGB")
    
    # ✅ OpenCV로 변환 후 전처리 (그레이스케일 변환 및 ROI 선택)
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ✅ 가장 큰 객체 영역 선택
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        margin = 20
        height, width, _ = img_cv.shape
        x_new, y_new = max(0, x - margin), max(0, y - margin)
        x_end, y_end = min(width, x + w + margin), min(height, y + h + margin)
        cropped = img_cv[y_new:y_end, x_new:x_end]
        resized_image = cv2.resize(cropped, (500, 500), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(resized_image)

    # ✅ 이미지 텐서 변환
    img_tensor = F.to_tensor(image).to(device)

    # ✅ 모델 예측 수행
    if model_option == "단일 모델":
        with torch.no_grad():
            pred = model([img_tensor])[0]
            # ✅ 바운딩 박스, 점수, 라벨 추출
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        # ✅ 신뢰도 임계값(0.5) 적용
        threshold = 0.5
        keep = scores > threshold
        final_boxes = boxes[keep]
        final_scores = scores[keep]
        final_labels = labels[keep]
    else:
        final_boxes_list, final_scores_list, final_labels_list = [], [], []
        for path in model_paths_kfold:
            model_kfold = load_model(path, num_classes=5, device=device)
            with torch.no_grad():
                pred = model_kfold([img_tensor])[0]

            final_boxes_list.append(pred["boxes"])
            final_scores_list.append(pred["scores"])
            final_labels_list.append(pred["labels"])

        final_boxes = torch.cat(final_boxes_list, dim=0)
        final_scores = torch.cat(final_scores_list, dim=0)
        final_labels = torch.cat(final_labels_list, dim=0)

        # ✅ NMS 적용
        keep = final_scores > 0.5
        final_boxes, final_scores, final_labels = final_boxes[keep], final_scores[keep], final_labels[keep]
        keep_indices = nms(final_boxes, final_scores, 0.3)
        final_boxes, final_scores, final_labels = final_boxes[keep_indices], final_scores[keep_indices], final_labels[keep_indices]

    # ✅ 결과 여부 출력
    result_text = "⚠ 결함 존재!" if len(final_boxes) > 0 else "✅ 결함 없음!"
    text_color = "red" if len(final_boxes) > 0 else "black"
    st.markdown(f"<h2 style='text-align: center; color: {text_color};'>{result_text}</h2>", unsafe_allow_html=True)

    # ✅ 결과 시각화
    fig, ax = plt.subplots(figsize=(8, 8))
    img_with_boxes = draw_bounding_boxes((img_tensor * 255).byte(), final_boxes, colors="red", width=3)
    ax.imshow(img_with_boxes.permute(1, 2, 0).cpu())

    for i in range(len(final_boxes)):
        x1, y1, _, _ = final_boxes[i]
        label = labels_inv.get(final_labels[i].item(), "Unknown")
        score = round(final_scores[i].item(), 2)
        ax.text(x1.item(), y1.item() - 5, f"{label} ({score:.2f})", color='white', fontsize=15, weight='bold', bbox=dict(facecolor='red', alpha=0.5))

    ax.axis("off")
    st.pyplot(fig)

    # ✅ 이전/다음 버튼
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.button("⬅ 이전", disabled=(st.session_state.image_index == 0), on_click=lambda: st.session_state.update({"image_index": st.session_state.image_index - 1}))
    with col3:
        st.button("다음 ➡", disabled=(st.session_state.image_index == num_files - 1), on_click=lambda: st.session_state.update({"image_index": st.session_state.image_index + 1}))

    # ✅ ZIP 다운로드 버튼
    if st.button("📂 결과 저장"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            zipf.writestr(f"results/{image_name}", img_tensor.cpu().numpy().tobytes())
        zip_buffer.seek(0)
        st.download_button(label="📥 ZIP 파일 다운로드", data=zip_buffer, file_name="detection_results.zip", mime="application/zip")
