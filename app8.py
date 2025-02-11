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
    if not os.path.exists(model_path):  # 모델이 없으면 다운로드
        os.makedirs("models", exist_ok=True)  # 폴더 생성
        gdown.download(drive_link, model_path, quiet=False)
    return model_path

# ✅ Google Drive 공유 링크 설정 (ID 변경 필요)
model_links = {
    "best_model.pth": "https://drive.google.com/uc?id=11u2cUNul_DZ0bJmykxXr90c9xYQD13CA",
    "best_model_fold_1.pth": "https://drive.google.com/uc?id=1ta9lx56Y74ypc87f-vx7fMlofdCS3O6V",
    "best_model_fold_2.pth": "https://drive.google.com/uc?id=1KzJLz-pMGbUjIVCLBH6WS_ezAsU-nLIi",
    "best_model_fold_3.pth": "https://drive.google.com/uc?id=1eN69P2RvHW7OrdJAHL6qgAML5_ArWXW3",
    "best_model_fold_4.pth": "https://drive.google.com/uc?id=10UNC7LV06GZlqnIvmG9S9F0KUEjLRQfW",
    "best_model_fold_5.pth": "https://drive.google.com/uc?id=11wGXOe6yZZgo8wDmzHrtFiJROn3JusKl",
}

# ✅ 훈련된 Faster R-CNN 모델 로드 함수
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

# ✅ 왼쪽 사이드바에 모델 선택 추가
st.sidebar.markdown("<h3 style='font-size:20px;'>🛠 사용할 모델을 선택하세요</h3>", unsafe_allow_html=True)
model_option = st.sidebar.selectbox("", ["단일 모델", "K-Fold 앙상블"])

# ✅ 선택한 모델 로드
if model_option == "단일 모델":
    model = load_model(model_path_single, num_classes=5, device=device)

# ✅ 왼쪽 사이드바에 이미지 업로드 추가
st.sidebar.markdown("<h3 style='font-size:20px;'>📂 이미지를 업로드하세요</h3>", unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# ✅ 이미지 인덱스 저장 (이전/다음 버튼 구현)
if "image_index" not in st.session_state:
    st.session_state.image_index = 0  # 첫 번째 이미지

if uploaded_files:
    num_files = len(uploaded_files)
    
    # ✅ 업로드된 이미지가 2개 이상일 때만 슬라이더 표시
    if num_files > 1:
        st.markdown("<h3 style='font-size:20px;'>📷 선택된 이미지</h3>", unsafe_allow_html=True)
        st.session_state.image_index = st.slider(
            "", 
            min_value=0, max_value=num_files-1, 
            value=st.session_state.image_index
        )
    else:
        st.session_state.image_index = 0  # 이미지가 1개뿐이면 0번째 이미지 선택
    
    # ✅ 현재 이미지 선택
    uploaded_file = uploaded_files[st.session_state.image_index]

    # ✅ 이미지 로드 및 전처리
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
    if model_option == "단일 모델":  # 단일 모델 사용
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

    else:  # ✅ K-Fold 모델 Lazy Loading 적용 (한 번에 로드 X, 하나씩 불러오기)
        final_boxes_list, final_scores_list, final_labels_list = [], [], []
        
        for path in model_paths_kfold:
            model_kfold = load_model(path, num_classes=5, device=device)  # 하나씩 로드
            with torch.no_grad():
                pred = model_kfold([img_tensor])[0]
            final_boxes_list.append(pred["boxes"])
            final_scores_list.append(pred["scores"])
            final_labels_list.append(pred["labels"])

        # ✅ 결과 병합
        final_boxes = torch.cat(final_boxes_list, dim=0)
        final_scores = torch.cat(final_scores_list, dim=0)
        final_labels = torch.cat(final_labels_list, dim=0)

        # ✅ NMS 적용
        iou_threshold, score_threshold = 0.3, 0.5
        keep = final_scores > score_threshold
        final_boxes, final_scores, final_labels = final_boxes[keep], final_scores[keep], final_labels[keep]
        keep_indices = nms(final_boxes, final_scores, iou_threshold)
        final_boxes, final_scores, final_labels = final_boxes[keep_indices], final_scores[keep_indices], final_labels[keep_indices]

    # ✅ 이전/다음 버튼 UI (비활성화 추가)
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        st.button("⬅ 이전", key="prev", disabled=(st.session_state.image_index == 0), 
                  on_click=lambda: st.session_state.update({"image_index": st.session_state.image_index - 1}))

    with col3:
        st.button("다음 ➡", key="next", disabled=(st.session_state.image_index == num_files - 1), 
                  on_click=lambda: st.session_state.update({"image_index": st.session_state.image_index + 1}))
                
    # ✅ 바운딩 박스가 있는지 여부 확인 후 색상 설정
    if len(final_boxes) > 0:
        result_text = "⚠ 결함 존재!"
        text_color = "red"
    else:
        result_text = "✅ 결함 없음!"
        text_color = "black"

    # ✅ 결함 여부 텍스트 출력 (가운데 정렬 + 색상 변경)
    st.markdown(
        f"<h2 style='text-align: center; color: {text_color};'>{result_text}</h2>",
        unsafe_allow_html=True
    )

    # ✅ 바운딩 박스 시각화
    img_with_boxes = draw_bounding_boxes(
        (img_tensor * 255).byte(), final_boxes, colors="red", width=3
    )

    # ✅ Matplotlib을 사용하여 시각화
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_with_boxes.permute(1, 2, 0).cpu())

    for i in range(len(final_boxes)):
        x1, y1, _, _ = final_boxes[i]
        label = labels_inv.get(final_labels[i].item(), "Unknown")
        score = final_scores[i].item()
        ax.text(x1.item(), y1.item() - 5, f"{label} ({score:.2f})",
                color='white', fontsize=15, weight='bold', bbox=dict(facecolor='red', alpha=0.5))

    ax.axis("off")
    st.pyplot(fig)

# ✅ 결과 이미지와 JSON 데이터 저장을 위한 리스트
all_results = []
all_images = []

# ✅ 이미지 다운로드 함수 (ZIP으로 저장)
def get_zip_download_link(zip_filename="results.zip"):
    """여러 개의 결과 파일을 ZIP으로 묶어 다운로드"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for img_name, img_pil in all_images:
            img_io = io.BytesIO()
            img_pil.save(img_io, format="JPEG")
            zipf.writestr(f"images/{img_name}", img_io.getvalue())

        # JSON 저장
        json_str = json.dumps(all_results, indent=4)
        zipf.writestr("results.json", json_str)

    zip_buffer.seek(0)
    b64 = base64.b64encode(zip_buffer.getvalue()).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}">📥 결과 ZIP 다운로드</a>'
    return href

# ✅ 모든 이미지 저장 버튼 UI
if st.button("💾 전체 결과 저장"):
    all_results.clear()
    all_images.clear()

    for idx, uploaded_file in enumerate(uploaded_files):
        # ✅ 이미지 로드
        image_name = uploaded_file.name  # 원본 이미지 파일 이름
        image = Image.open(uploaded_file).convert("RGB")

        # ✅ 이미지 처리 (바운딩 박스 추가)
        img_with_boxes = draw_bounding_boxes(
            (F.to_tensor(image) * 255).byte(), final_boxes, colors="red", width=3
        )

        img_pil = Image.fromarray(img_with_boxes.permute(1, 2, 0).byte().cpu().numpy())

        # ✅ JSON 데이터 저장
        result_data = {
            "image_name": image_name,
            "num_detections": len(final_boxes),
            "detections": []
        }

        for i in range(len(final_boxes)):
            x1, y1, x2, y2 = final_boxes[i].tolist()
            label = labels_inv.get(final_labels[i].item(), "Unknown")
            score = round(final_scores[i].item(), 2)
            
            result_data["detections"].append({
                "label": label,
                "confidence": score,
                "bbox": [x1, y1, x2, y2]
            })

        all_results.append(result_data)
        all_images.append((image_name, img_pil))

    # ✅ ZIP 다운로드 링크 생성
    st.markdown(get_zip_download_link(), unsafe_allow_html=True)
