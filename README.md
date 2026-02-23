# 8-Camera 360° Panorama Stitching System

실시간 360도 파노라마 스티칭 시스템 - 8대의 카메라로부터 UDP를 통해 프레임을 수신하여 파노라마 이미지를 생성하고 성능을 평가합니다.

## 📋 실행 명령어

### 기본 실행 (1회 테스트)
python3 collection_rate_test_v2.py \
    --calibration_dir ./calibration_images \
    --calibration_frames 10 \
    --reference_frame 7 \
    --ports 5001 5002 \
    --camera_order 5 4 3 2 1 8 7 6 \
    --scale 1.0 \
    --num_trials 1 \
    --output_dir ./test/collection_test_1

### 5회 반복 테스트
python3 collection_rate_test_v2.py \
    --calibration_dir ./calibration_images \
    --calibration_frames 10 \
    --reference_frame 7 \
    --ports 5001 5002 \
    --camera_order 5 4 3 2 1 8 7 6 \
    --scale 1.0 \
    --num_trials 5 \
    --trial_interval 10 \
    --output_dir ./output/collection_test

## 🖥️ 시스템 요구사항

- OS: Ubuntu 22.04.5 LTS
- Python: 3.10.12
- OpenCV: 4.10.0 (CUDA 지원)
- NumPy: 1.24.3
- GPU: NVIDIA GeForce GTX 1050 Ti (4GB)
- CUDA: 11.8

## 📷 카메라 설정

- 카메라 수: 8대
- 해상도: 1016x760 픽셀 (각 카메라)
- 전송 해상도: 2032x1520 픽셀 (라즈베리파이당)
- 카메라 순서: [5, 4, 3, 2, 1, 8, 7, 6]
- 참조 프레임: 7번 카메라

## 📁 프로젝트 구조

camera_calibration_project/
├── README.md
├── collection_rate_test_v2.py         # 메인 스크립트
├── collection_rate_test_v3_stable.py  # 안정적 출력 버전
└── calibration_images/                # 캘리브레이션 이미지

## 🔧 설치 방법

# OpenCV 설치
pip3 install opencv-python opencv-contrib-python

# NumPy 설치
pip3 install numpy

## 📊 출력 결과

- 파노라마 이미지: pano_0001.jpg, pano_0002.jpg, ...
- 성능 CSV: trial_N_result.csv
- 요약 CSV: summary.csv

## 🎯 성능 목표

- 처리 속도: ~1.2 FPS (826ms/frame)
- 수집률: 90% 이상

## 📞 연락처

- GitHub: https://github.com/tophtud/8camera-panorama-stitching
