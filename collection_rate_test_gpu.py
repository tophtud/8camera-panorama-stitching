#!/usr/bin/env python3
"""
영상 수집율 시험 평가 (CuPy GPU 가속)

기존 CuPy GPU 스티칭 코드 기반 + UDP 실시간 입력 + 수집율 시험

구조:
  Step 1: 캘리브레이션 (폴더 이미지 or NPZ 로드)
    - OpenCV Stitcher → cameras() → 워핑 맵 생성
    - CuPy GPU 메모리 업로드
    - NPZ 저장
  
  Step 2: 영상 수집율 시험 (UDP 실시간)
    - 2분간 1초당 1장 파노라마 생성 (목표: 120장)
    - 5회 반복 실시
    - CSV에 생성 시간 기록
    - 화면 표시 없음 (성능 최적화)
    - 목표 수집율: 90%

사용법:
  # 캘리브레이션 + 시험
  python3 collection_rate_test_gpu.py \
      --calibration_dir ./calibration_images \
      --reference_frame 7 \
      --camera_order 5 4 3 2 1 8 7 6 \
      --ports 5001 5002 \
      --scale 1.0 \
      --num_trials 5 \
      --output_dir ./output/collection_test

  # NPZ 로드 + 시험 (캘리브레이션 스킵)
  python3 collection_rate_test_gpu.py \
      --calibration_npz ./calibration_gpu.npz \
      --ports 5001 5002 \
      --num_trials 5 \
      --output_dir ./output/collection_test
"""

import os
import glob
import time
import argparse
import csv
import numpy as np
import cv2
import threading
from queue import Queue, Empty
from datetime import datetime

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    if CUPY_AVAILABLE:
        print(f"[✅ CuPy GPU] {cp.cuda.runtime.getDeviceCount()}개 GPU, 버전: {cp.__version__}")
except ImportError:
    CUPY_AVAILABLE = False
    print("[❌ CuPy 없음] CPU 모드로 실행")


# ============================================================
# GPU 스티칭 엔진 (기존 작동 코드 기반)
# ============================================================
class StitcherGPU:
    """OpenCV Stitcher 캘리브레이션 + CuPy GPU 워핑/블렌딩"""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.use_gpu = CUPY_AVAILABLE
        self.cameras = None
        self.avg_focal = None
        self.gpu_xmaps = None
        self.gpu_ymaps = None
        self.gpu_masks = None
        self.corners = None
        self.xmaps, self.ymaps, self.sizes = [], [], []
        self.pano_size = None
        self.pano_offset = None
        self.input_size = None
        if self.use_gpu:
            _ = cp.array([1])
            cp.cuda.Stream.null.synchronize()

    def calibrate(self, images, save_path=None):
        """OpenCV Stitcher로 캘리브레이션 → 워핑 맵 생성"""
        print("=" * 60)
        print(f"캘리브레이션 ({int(self.scale * 100)}% 해상도)")
        print("=" * 60)

        images_scaled = [cv2.resize(img, None, fx=self.scale, fy=self.scale) for img in images]
        h, w = images_scaled[0].shape[:2]
        self.input_size = (w, h)
        print(f"  입력: {len(images_scaled)}개 이미지, 각 {w}x{h}")

        # 1. OpenCV Stitcher 실행
        print("\n[1/3] OpenCV Stitcher 실행...")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        stitcher.setRegistrationResol(0.6)
        stitcher.setSeamEstimationResol(0.1)
        stitcher.setCompositingResol(-1)
        stitcher.setPanoConfidenceThresh(0.5)

        status = stitcher.estimateTransform(images_scaled)
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 변환 추정 실패: {status}")
            return False
        print("  ✅ 변환 추정 성공!")

        # 2. 카메라 파라미터 추출
        print("\n[2/3] 카메라 파라미터 추출...")
        self.cameras = stitcher.cameras()
        self.avg_focal = np.mean([cam.focal for cam in self.cameras])
        print(f"  카메라: {len(self.cameras)}개, 평균 focal: {self.avg_focal:.2f}")

        # 기준 파노라마 생성
        status2, reference = stitcher.composePanorama()
        if status2 != cv2.Stitcher_OK:
            print(f"  ❌ 파노라마 생성 실패: {status2}")
            return False
        self.pano_size = (reference.shape[1], reference.shape[0])
        print(f"  ✅ 파노라마: {self.pano_size[0]}x{self.pano_size[1]}")

        # 3. 워핑 맵 생성
        print("\n[3/3] 워핑 맵 생성...")
        self._build_warp_maps()

        # GPU 업로드
        if self.use_gpu:
            print("\n[GPU] 메모리 업로드...")
            self._upload_to_gpu()

        # NPZ 저장
        if save_path:
            self._save_calibration(save_path, reference)

        print("\n✅ 캘리브레이션 완료!")
        return True, reference

    def _build_warp_maps(self):
        """SphericalWarper로 워핑 맵 생성"""
        w, h = self.input_size
        warper = cv2.PyRotationWarper('spherical', self.avg_focal)
        K = np.array([
            [self.avg_focal, 0, w / 2],
            [0, self.avg_focal, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        self.xmaps, self.ymaps, self.corners, self.sizes = [], [], [], []

        for i, cam in enumerate(self.cameras):
            R = cam.R.astype(np.float32)
            roi, xmap, ymap = warper.buildMaps((w, h), K, R)
            self.corners.append((roi[0], roi[1]))
            self.sizes.append((roi[2], roi[3]))
            self.xmaps.append(xmap)
            self.ymaps.append(ymap)
            print(f"  카메라 {i + 1}: corner=({roi[0]},{roi[1]}), size=({roi[2]},{roi[3]})")

        min_x = min(c[0] for c in self.corners)
        min_y = min(c[1] for c in self.corners)
        max_x = max(c[0] + s[0] for c, s in zip(self.corners, self.sizes))
        max_y = max(c[1] + s[1] for c, s in zip(self.corners, self.sizes))
        self.pano_offset = (min_x, min_y)
        self.pano_size = (max_x - min_x, max_y - min_y)
        print(f"  파노라마: {self.pano_size[0]}x{self.pano_size[1]}, offset=({min_x},{min_y})")

    def _upload_to_gpu(self):
        """워핑 맵과 마스크를 GPU 메모리에 업로드"""
        self.gpu_xmaps = [cp.asarray(x) for x in self.xmaps]
        self.gpu_ymaps = [cp.asarray(y) for y in self.ymaps]

        w, h = self.input_size
        self.gpu_masks = []
        for xmap, ymap in zip(self.xmaps, self.ymaps):
            valid = (xmap >= 0) & (xmap < w) & (ymap >= 0) & (ymap < h)
            self.gpu_masks.append(cp.asarray(valid.astype(np.float32)))

        print(f"  GPU 메모리: {cp.get_default_memory_pool().used_bytes() / 1024 ** 2:.1f} MB")

    def _save_calibration(self, path, reference):
        """캘리브레이션 결과를 NPZ로 저장"""
        save_dict = {
            'num_views': len(self.cameras),
            'input_size': np.array(self.input_size),
            'pano_size': np.array(self.pano_size),
            'pano_offset': np.array(self.pano_offset),
            'scale': self.scale,
            'avg_focal': self.avg_focal
        }
        for i, cam in enumerate(self.cameras):
            save_dict[f'R_{i}'] = cam.R
            save_dict[f'xmap_{i}'] = self.xmaps[i]
            save_dict[f'ymap_{i}'] = self.ymaps[i]
            save_dict[f'corner_{i}'] = np.array(self.corners[i])
            save_dict[f'size_{i}'] = np.array(self.sizes[i])

        np.savez_compressed(path, **save_dict)
        ref_path = path.replace('.npz', '_reference.jpg')
        cv2.imwrite(ref_path, reference)
        print(f"  ✅ NPZ 저장: {path}")
        print(f"  ✅ 기준 파노라마 저장: {ref_path}")

    def load_calibration(self, path):
        """NPZ에서 캘리브레이션 로드"""
        print(f"\n[NPZ] 캘리브레이션 로드: {path}")
        data = np.load(path)
        num_views = int(data['num_views'])
        self.input_size = tuple(data['input_size'])
        self.pano_size = tuple(data['pano_size'])
        self.pano_offset = tuple(data['pano_offset'])
        self.scale = float(data['scale'])
        self.avg_focal = float(data['avg_focal'])

        self.xmaps, self.ymaps, self.corners, self.sizes = [], [], [], []
        for i in range(num_views):
            self.xmaps.append(data[f'xmap_{i}'])
            self.ymaps.append(data[f'ymap_{i}'])
            self.corners.append(tuple(data[f'corner_{i}']))
            self.sizes.append(tuple(data[f'size_{i}']))

        print(f"  뷰: {num_views}개, 스케일: {self.scale}, 파노라마: {self.pano_size}")

        if self.use_gpu:
            self._upload_to_gpu()

        return True

    def stitch_gpu(self, images):
        """CuPy GPU로 워핑 + 블렌딩"""
        images_scaled = [cv2.resize(img, None, fx=self.scale, fy=self.scale) for img in images]
        pano_w, pano_h = int(self.pano_size[0]), int(self.pano_size[1])
        offset_x, offset_y = int(self.pano_offset[0]), int(self.pano_offset[1])

        result = cp.zeros((pano_h, pano_w, 3), dtype=cp.float32)
        weight_sum = cp.zeros((pano_h, pano_w), dtype=cp.float32)

        w, h = int(self.input_size[0]), int(self.input_size[1])

        for img, xmap, ymap, mask, corner, size in zip(
                images_scaled, self.gpu_xmaps, self.gpu_ymaps,
                self.gpu_masks, self.corners, self.sizes):

            img_gpu = cp.asarray(img.astype(np.float32))

            # 쌍선형 보간
            x0 = cp.floor(xmap).astype(cp.int32)
            y0 = cp.floor(ymap).astype(cp.int32)
            x1, y1 = x0 + 1, y0 + 1
            x0, x1 = cp.clip(x0, 0, w - 1), cp.clip(x1, 0, w - 1)
            y0, y1 = cp.clip(y0, 0, h - 1), cp.clip(y1, 0, h - 1)

            wx = xmap - cp.floor(xmap)
            wy = ymap - cp.floor(ymap)

            p00 = img_gpu[y0, x0]
            p01 = img_gpu[y0, x1]
            p10 = img_gpu[y1, x0]
            p11 = img_gpu[y1, x1]

            wx = wx[:, :, cp.newaxis]
            wy = wy[:, :, cp.newaxis]

            warped = p00 * (1 - wx) * (1 - wy) + p01 * wx * (1 - wy) + \
                     p10 * (1 - wx) * wy + p11 * wx * wy

            # 파노라마에 배치
            dst_x = int(corner[0]) - offset_x
            dst_y = int(corner[1]) - offset_y
            wh, ww = warped.shape[:2]

            src_x1, src_y1, src_x2, src_y2 = 0, 0, ww, wh
            if dst_x < 0:
                src_x1 = -dst_x
                dst_x = 0
            if dst_y < 0:
                src_y1 = -dst_y
                dst_y = 0
            if dst_x + (src_x2 - src_x1) > pano_w:
                src_x2 = src_x1 + (pano_w - dst_x)
            if dst_y + (src_y2 - src_y1) > pano_h:
                src_y2 = src_y1 + (pano_h - dst_y)

            # 가우시안 블렌딩 가중치
            center_x = (src_x2 + src_x1) / 2
            center_y = (src_y2 + src_y1) / 2
            yy, xx = cp.mgrid[src_y1:src_y2, src_x1:src_x2]
            dist = cp.sqrt(((xx - center_x) / (ww / 2)) ** 2 + ((yy - center_y) / (wh / 2)) ** 2)
            blend_weight = cp.exp(-2 * dist ** 2) * mask[src_y1:src_y2, src_x1:src_x2]

            dst_h = src_y2 - src_y1
            dst_w = src_x2 - src_x1

            result[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w] += \
                warped[src_y1:src_y2, src_x1:src_x2] * blend_weight[:, :, cp.newaxis]
            weight_sum[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w] += blend_weight

        weight_sum = cp.maximum(weight_sum, 1e-6)
        result = result / weight_sum[:, :, cp.newaxis]

        cp.cuda.Stream.null.synchronize()
        return np.clip(cp.asnumpy(result), 0, 255).astype(np.uint8)

    def stitch_cpu(self, images):
        """CPU 폴백 (OpenCV remap)"""
        images_scaled = [cv2.resize(img, None, fx=self.scale, fy=self.scale) for img in images]
        pano_w, pano_h = int(self.pano_size[0]), int(self.pano_size[1])
        offset_x, offset_y = int(self.pano_offset[0]), int(self.pano_offset[1])

        result = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        weight_sum = np.zeros((pano_h, pano_w), dtype=np.float32)

        for img, xmap, ymap, corner, size in zip(
                images_scaled, self.xmaps, self.ymaps, self.corners, self.sizes):

            warped = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)

            dst_x = int(corner[0]) - offset_x
            dst_y = int(corner[1]) - offset_y
            wh, ww = warped.shape[:2]

            src_x1, src_y1, src_x2, src_y2 = 0, 0, ww, wh
            if dst_x < 0:
                src_x1 = -dst_x
                dst_x = 0
            if dst_y < 0:
                src_y1 = -dst_y
                dst_y = 0
            if dst_x + (src_x2 - src_x1) > pano_w:
                src_x2 = src_x1 + (pano_w - dst_x)
            if dst_y + (src_y2 - src_y1) > pano_h:
                src_y2 = src_y1 + (pano_h - dst_y)

            mask = np.any(warped[src_y1:src_y2, src_x1:src_x2] > 0, axis=2).astype(np.float32)
            dst_h = src_y2 - src_y1
            dst_w = src_x2 - src_x1

            result[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w] += \
                warped[src_y1:src_y2, src_x1:src_x2].astype(np.float32) * mask[:, :, np.newaxis]
            weight_sum[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w] += mask

        weight_sum = np.maximum(weight_sum, 1e-6)
        result = result / weight_sum[:, :, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)

    def stitch(self, images):
        """자동 GPU/CPU 선택"""
        if self.use_gpu:
            return self.stitch_gpu(images)
        else:
            return self.stitch_cpu(images)


# ============================================================
# UDP 수신기
# ============================================================
class UDPReceiver:
    """UDP 스트림 수신 (멀티스레드)"""

    def __init__(self, port, name="Camera"):
        self.port = port
        self.name = name
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"  {self.name}: UDP 포트 {self.port} 수신 시작...")

    def _receive_loop(self):
        pipeline = (
            f"udpsrc port={self.port} "
            f"caps=\"application/x-rtp, media=video, clock-rate=90000, "
            f"encoding-name=JPEG, payload=96\" ! "
            f"rtpjpegdepay ! jpegdec ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink sync=false drop=true max-buffers=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(f"  ❌ {self.name}: 스트림 열기 실패!")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put(frame)

        cap.release()

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)


# ============================================================
# 영상 수집율 시험
# ============================================================
class CollectionRateTest:
    """영상 수집율 시험 평가"""

    def __init__(self, args):
        self.args = args
        self.stitcher = StitcherGPU(scale=args.scale)
        self.receivers = []

    def calibrate_from_folder(self):
        """폴더 이미지로 캘리브레이션"""
        print("\n" + "=" * 60)
        print("폴더 이미지로 캘리브레이션")
        print("=" * 60)

        camera_order = self.args.camera_order
        cal_dir = os.path.expanduser(self.args.calibration_dir)

        print(f"  카메라 순서: {camera_order}")

        all_images = []
        for cam_idx in camera_order:
            cam_dir = os.path.join(cal_dir, f'MyCam_{cam_idx:03d}')
            if not os.path.exists(cam_dir):
                print(f"  ⚠️ 카메라 {cam_idx} 디렉토리 없음: {cam_dir}")
                continue
            img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
            if not img_files:
                img_files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))
            if not img_files:
                print(f"  ⚠️ 카메라 {cam_idx} 이미지 없음")
                continue
            images = []
            for img_file in img_files[:self.args.calibration_frames]:
                img = cv2.imread(img_file)
                if img is not None:
                    images.append(img)
            if images:
                all_images.append(images)
                print(f"  ✅ 카메라 {cam_idx}: {len(images)}장")

        if len(all_images) < 2:
            print("\n❌ 카메라가 부족합니다")
            return False

        ref_idx = min(self.args.reference_frame - 1, len(all_images[0]) - 1)
        ref_idx = max(0, ref_idx)
        ref_images = [cam_images[ref_idx] for cam_images in all_images]
        print(f"\n  기준 프레임: {ref_idx + 1}/{len(all_images[0])}")

        npz_path = os.path.join(self.args.output_dir, "calibration.npz")
        os.makedirs(self.args.output_dir, exist_ok=True)

        result = self.stitcher.calibrate(ref_images, save_path=npz_path)
        if result is False:
            return False

        return True

    def init_receivers(self):
        """UDP 수신기 초기화"""
        print("\n" + "=" * 60)
        print("UDP 수신기 초기화")
        print("=" * 60)
        for i, port in enumerate(self.args.ports):
            receiver = UDPReceiver(port, f"UDP_{i + 1}")
            receiver.start()
            self.receivers.append(receiver)
        print("  프레임 수신 대기 중...")
        time.sleep(3)

    def stop_receivers(self):
        """UDP 수신기 정지"""
        for receiver in self.receivers:
            receiver.stop()
        self.receivers = []

    def split_2x2(self, frame):
        """2x2 분할"""
        h, w = frame.shape[:2]
        h2, w2 = h // 2, w // 2
        return [
            frame[0:h2, 0:w2],
            frame[0:h2, w2:w],
            frame[h2:h, 0:w2],
            frame[h2:h, w2:w]
        ]

    def get_8cam_images(self):
        """UDP에서 8개 카메라 이미지 추출"""
        frames = []
        for receiver in self.receivers:
            frame = receiver.get_frame(timeout=1.0)
            if frame is None:
                return None
            frames.append(frame)

        # 2x2 분할 → 8개 이미지
        images_8cam = []
        for frame in frames:
            images_8cam.extend(self.split_2x2(frame))

        # 카메라 순서 재배치
        if self.args.camera_order:
            reordered = [images_8cam[i - 1] for i in self.args.camera_order]
            return reordered

        return images_8cam

    def warmup_gpu(self):
        """GPU 워밍업 (첫 실행 지연 방지)"""
        if not CUPY_AVAILABLE:
            return
        print("\n[GPU] 워밍업 중...")
        # 더미 이미지로 3회 실행
        dummy_images = [np.random.randint(0, 255, (760, 1016, 3), dtype=np.uint8)
                        for _ in range(len(self.args.camera_order))]
        for i in range(3):
            try:
                _ = self.stitcher.stitch(dummy_images)
            except Exception:
                pass
        cp.cuda.Stream.null.synchronize()
        print("  ✅ GPU 워밍업 완료")

    def run_single_test(self, trial_num, output_dir):
        """단일 시험 실행 (2분)"""
        test_duration = 120  # 2분 = 120초
        target_interval = 1.0  # 1초당 1장
        target_total = test_duration  # 120장 목표

        trial_dir = os.path.join(output_dir, f"trial_{trial_num}")
        os.makedirs(trial_dir, exist_ok=True)

        csv_path = os.path.join(trial_dir, f"trial_{trial_num}_result.csv")

        print(f"\n{'=' * 60}")
        print(f"[시험 {trial_num}/{self.args.num_trials}] 시작")
        print(f"{'=' * 60}")
        print(f"  목표: {test_duration}초 동안 {target_total}장 생성")
        print(f"  간격: {target_interval}초")
        print(f"  모드: {'GPU' if self.stitcher.use_gpu else 'CPU'}")
        print(f"  저장: {trial_dir}")
        print()

        # CSV 파일 생성
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'Frame_Number',       # 프레임 번호
            'Target_Time_s',      # 목표 시간 (초)
            'Actual_Time_s',      # 실제 경과 시간 (초)
            'Generation_Time',    # 생성 시각 (HH:MM:SS.mmm)
            'Processing_ms',      # 스티칭 처리 시간 (ms)
            'Total_ms',           # 전체 처리 시간 (ms, 수신+스티칭+저장)
            'Success',            # 성공 여부
            'Image_File'          # 이미지 파일명
        ])

        success_count = 0
        fail_count = 0
        total_processing_time = 0

        test_start_time = time.time()
        next_capture_time = test_start_time

        frame_num = 0

        while True:
            current_time = time.time()
            elapsed = current_time - test_start_time

            # 2분 경과 시 종료
            if elapsed >= test_duration:
                print(f"\n  ⏱️ {test_duration}초 경과 → 시험 종료")
                break

            # 1초 간격 타이머
            if current_time < next_capture_time:
                time.sleep(0.001)
                continue

            frame_num += 1
            target_time_s = (frame_num - 1) * target_interval
            actual_time_s = elapsed

            frame_start = time.time()

            # 1. 프레임 수신
            images = self.get_8cam_images()

            if images is None:
                fail_count += 1
                gen_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                csv_writer.writerow([
                    frame_num, f"{target_time_s:.1f}", f"{actual_time_s:.3f}",
                    gen_time, 0, 0, False, ""
                ])
                next_capture_time += target_interval
                continue

            # 2. GPU 스티칭
            stitch_start = time.time()
            try:
                pano = self.stitcher.stitch(images)
            except Exception as e:
                pano = None
                print(f"  ⚠️ 스티칭 오류: {e}")

            processing_ms = (time.time() - stitch_start) * 1000

            gen_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            if pano is not None:
                # 3. 이미지 저장
                success_count += 1
                img_filename = f"pano_{frame_num:04d}.jpg"
                img_path = os.path.join(trial_dir, img_filename)
                cv2.imwrite(img_path, pano, [cv2.IMWRITE_JPEG_QUALITY, 85])

                total_ms = (time.time() - frame_start) * 1000
                total_processing_time += processing_ms

                csv_writer.writerow([
                    frame_num, f"{target_time_s:.1f}", f"{actual_time_s:.3f}",
                    gen_time, f"{processing_ms:.1f}", f"{total_ms:.1f}",
                    True, img_filename
                ])
            else:
                fail_count += 1
                total_ms = (time.time() - frame_start) * 1000
                csv_writer.writerow([
                    frame_num, f"{target_time_s:.1f}", f"{actual_time_s:.3f}",
                    gen_time, f"{processing_ms:.1f}", f"{total_ms:.1f}",
                    False, ""
                ])

            # 진행 상황 (10초마다)
            if frame_num % 10 == 0:
                rate = (success_count / frame_num) * 100 if frame_num > 0 else 0
                avg_proc = total_processing_time / success_count if success_count > 0 else 0
                print(f"  [{elapsed:.0f}s] 프레임 {frame_num} | "
                      f"성공: {success_count} | 실패: {fail_count} | "
                      f"수집율: {rate:.1f}% | 평균: {avg_proc:.0f}ms")

            # 다음 캡처 시간
            next_capture_time += target_interval

            # 밀린 프레임 처리
            now = time.time()
            while next_capture_time + target_interval < now and (now - test_start_time) < test_duration:
                frame_num += 1
                fail_count += 1
                skip_target = (frame_num - 1) * target_interval
                skip_gen_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                csv_writer.writerow([
                    frame_num, f"{skip_target:.1f}",
                    f"{now - test_start_time:.3f}",
                    skip_gen_time, 0, 0, False, "SKIPPED"
                ])
                next_capture_time += target_interval

        csv_file.close()

        # 시험 결과 요약
        total_frames = success_count + fail_count
        collection_rate = (success_count / target_total) * 100 if target_total > 0 else 0
        avg_processing = total_processing_time / success_count if success_count > 0 else 0

        print(f"\n  {'=' * 40}")
        print(f"  [시험 {trial_num}] 결과")
        print(f"  {'=' * 40}")
        print(f"  목표 프레임: {target_total}장")
        print(f"  시도 프레임: {total_frames}장")
        print(f"  성공 프레임: {success_count}장")
        print(f"  실패 프레임: {fail_count}장")
        print(f"  수집율: {collection_rate:.1f}%")
        print(f"  평균 처리 시간: {avg_processing:.1f} ms")
        if collection_rate >= 90:
            print(f"  ✅ 목표 달성! (90% 이상)")
        else:
            print(f"  ❌ 목표 미달성 (90% 미만)")
        print(f"  CSV: {csv_path}")

        return {
            'trial': trial_num,
            'target_total': target_total,
            'total_frames': total_frames,
            'success': success_count,
            'fail': fail_count,
            'collection_rate': collection_rate,
            'avg_processing_ms': avg_processing,
            'csv_path': csv_path
        }

    def run(self):
        """전체 시험 실행"""
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1단계: 캘리브레이션
        if self.args.calibration_npz:
            self.stitcher.load_calibration(self.args.calibration_npz)
        elif self.args.calibration_dir:
            if not self.calibrate_from_folder():
                print("\n❌ 캘리브레이션 실패!")
                return
        else:
            print("\n❌ --calibration_dir 또는 --calibration_npz 필요!")
            return

        # 2단계: UDP 수신기 초기화
        self.init_receivers()

        # 3단계: GPU 워밍업
        self.warmup_gpu()

        # 4단계: 시험 실행
        print("\n" + "=" * 60)
        print("영상 수집율 시험 시작")
        print("=" * 60)
        print(f"  시험 횟수: {self.args.num_trials}회")
        print(f"  시험 시간: 120초 (2분)")
        print(f"  목표 수집율: 90%")
        print(f"  목표 프레임: 1초당 1장 (120장/2분)")
        print(f"  모드: {'CuPy GPU' if self.stitcher.use_gpu else 'CPU'}")
        print(f"  스케일: {self.args.scale * 100:.0f}%")
        print(f"  출력: {output_dir}")

        all_results = []

        for trial in range(1, self.args.num_trials + 1):
            if trial > 1:
                wait_sec = self.args.trial_interval
                print(f"\n⏳ 다음 시험까지 {wait_sec}초 대기...")
                time.sleep(wait_sec)

            result = self.run_single_test(trial, output_dir)
            all_results.append(result)

        # 수신기 정지
        self.stop_receivers()

        # 전체 결과 요약 CSV
        summary_csv = os.path.join(output_dir, "summary.csv")
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Trial', 'Target_Total', 'Total_Frames', 'Success',
                'Fail', 'Collection_Rate_%', 'Avg_Processing_ms'
            ])
            for r in all_results:
                writer.writerow([
                    r['trial'], r['target_total'], r['total_frames'],
                    r['success'], r['fail'],
                    f"{r['collection_rate']:.1f}",
                    f"{r['avg_processing_ms']:.1f}"
                ])

            avg_rate = np.mean([r['collection_rate'] for r in all_results])
            avg_proc = np.mean([r['avg_processing_ms'] for r in all_results])
            writer.writerow([])
            writer.writerow(['Average', '', '', '', '',
                             f"{avg_rate:.1f}", f"{avg_proc:.1f}"])

        # 최종 결과 출력
        print("\n" + "=" * 60)
        print("전체 시험 결과 요약")
        print("=" * 60)
        print(f"{'시험':>6} | {'목표':>6} | {'성공':>6} | {'실패':>6} | "
              f"{'수집율':>8} | {'처리시간':>10}")
        print("-" * 60)
        for r in all_results:
            status = "✅" if r['collection_rate'] >= 90 else "❌"
            print(f"  {r['trial']:>4} | {r['target_total']:>5}장 | "
                  f"{r['success']:>5}장 | {r['fail']:>5}장 | "
                  f"{r['collection_rate']:>6.1f}% {status} | "
                  f"{r['avg_processing_ms']:>8.1f}ms")
        print("-" * 60)
        avg_rate = np.mean([r['collection_rate'] for r in all_results])
        avg_proc = np.mean([r['avg_processing_ms'] for r in all_results])
        overall_status = "✅" if avg_rate >= 90 else "❌"
        print(f"  평균 |       |       |       | "
              f"{avg_rate:>6.1f}% {overall_status} | {avg_proc:>8.1f}ms")
        print("=" * 60)
        print(f"\n✅ 요약 CSV: {summary_csv}")
        print(f"✅ 각 시험 결과: {output_dir}/trial_N/trial_N_result.csv")


def main():
    parser = argparse.ArgumentParser(description="영상 수집율 시험 평가 (CuPy GPU 가속)")

    # 캘리브레이션 설정
    parser.add_argument("--calibration_dir", type=str,
                        help="캘리브레이션용 폴더 경로")
    parser.add_argument("--calibration_npz", type=str,
                        help="기존 캘리브레이션 NPZ 파일 (캘리브레이션 스킵)")
    parser.add_argument("--calibration_frames", type=int, default=10)
    parser.add_argument("--reference_frame", type=int, default=7)

    # UDP 설정
    parser.add_argument("--ports", type=int, nargs="+", default=[5001, 5002])

    # 카메라 설정
    parser.add_argument("--camera_order", type=int, nargs="+",
                        default=[5, 4, 3, 2, 1, 8, 7, 6])

    # 스티칭 설정
    parser.add_argument("--scale", type=float, default=1.0)

    # 시험 설정
    parser.add_argument("--num_trials", type=int, default=5,
                        help="시험 횟수 (기본: 5)")
    parser.add_argument("--trial_interval", type=int, default=10,
                        help="시험 간 대기 시간 (초, 기본: 10)")

    # 저장 설정
    parser.add_argument("--output_dir", type=str,
                        default="./output/collection_test",
                        help="결과 저장 디렉토리")

    args = parser.parse_args()

    print("=" * 60)
    print("영상 수집율 시험 평가 (CuPy GPU 가속)")
    print("=" * 60)
    if args.calibration_npz:
        print(f"캘리브레이션 NPZ: {args.calibration_npz}")
    else:
        print(f"캘리브레이션 폴더: {args.calibration_dir}")
        print(f"기준 프레임: {args.reference_frame}")
    print(f"UDP 포트: {args.ports}")
    print(f"스케일: {args.scale * 100:.0f}%")
    print(f"카메라 순서: {args.camera_order}")
    print(f"시험 횟수: {args.num_trials}회")
    print(f"시험 시간: 120초 (2분)")
    print(f"목표: 1초당 1장 (120장/2분)")
    print(f"목표 수집율: 90%")
    print(f"GPU: {'CuPy' if CUPY_AVAILABLE else 'CPU (폴백)'}")
    print(f"출력: {args.output_dir}")
    print("=" * 60)

    tester = CollectionRateTest(args)
    tester.run()


if __name__ == "__main__":
    main()
