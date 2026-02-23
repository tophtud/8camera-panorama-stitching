#!/usr/bin/env python3
"""
CuPy GPU 파노라마 스티칭 v6 + CSV 성능 로깅 (안정화 버전)
- 특징점 매칭 기반
- CuPy GPU 가속 워핑/블렌딩
- UDP 수신 + 실시간 파노라마
- 저장 속도/처리 속도 CSV 기록
- cp.einsum 제거 (CUBLAS_STATUS_NOT_SUPPORTED 회피)
- 기준 파노라마 생성 생략 옵션 제공 (OOM/CUBLAS 방지)
"""

import os
import glob
import argparse
import time
import csv
from datetime import datetime
import numpy as np
import cv2
import threading
from queue import Queue, Empty

# CuPy 확인
try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    if CUPY_AVAILABLE:
        print(f"[✅ GPU] CuPy {cp.__version__} | {cp.cuda.runtime.getDeviceCount()}개 GPU 활성화")
        _ = cp.array([1], dtype=cp.float32)  # GPU 워밍업
        cp.cuda.Stream.null.synchronize()
    else:
        print("[❌ GPU] GPU를 찾을 수 없습니다")
        raise SystemExit(1)
except ImportError:
    print("[❌ GPU] CuPy가 설치되지 않았습니다")
    print("설치 예시: pip3 install cupy-cuda11x")
    raise SystemExit(1)


class UDPReceiver:
    """UDP 스트림 수신"""
    def __init__(self, port, name="Camera"):
        self.port = port
        self.name = name
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"  {self.name}: UDP 포트 {self.port}")

    def _receive_loop(self):
        pipeline = (
            f"udpsrc port={self.port} "
            f"caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=JPEG, payload=96\" ! "
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

    def get_frame(self, timeout=0.1):
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


def split_2x2(frame):
    """2x2 분할"""
    h, w = frame.shape[:2]
    h2, w2 = h // 2, w // 2
    return [
        frame[0:h2, 0:w2],
        frame[0:h2, w2:w],
        frame[h2:h, 0:w2],
        frame[h2:h, w2:w]
    ]


def crop_edges(images, crop_config):
    """가장자리 크롭"""
    if crop_config is None:
        return images

    if isinstance(crop_config, int):
        c = crop_config
        if c <= 0:
            return images
        cropped = []
        for img in images:
            h, w = img.shape[:2]
            if (2 * c) >= h or (2 * c) >= w:
                cropped.append(img)  # 과도한 crop 방지
            else:
                cropped.append(img[c:-c, c:-c])
        return cropped

    cl, cr, ct, cb = crop_config
    cropped = []
    for img in images:
        h, w = img.shape[:2]
        x1 = max(0, cl)
        x2 = min(w, w - cr if cr > 0 else w)
        y1 = max(0, ct)
        y2 = min(h, h - cb if cb > 0 else h)
        if x1 >= x2 or y1 >= y2:
            cropped.append(img)
        else:
            cropped.append(img[y1:y2, x1:x2])
    return cropped


def load_camera_images(input_dir, num_frames=10, crop_config=None, camera_order=None):
    """폴더에서 이미지 로드"""
    print("\n[폴더] 캘리브레이션 이미지 로드")

    if camera_order is None:
        camera_order = list(range(1, 9))

    print(f"  카메라 순서: {camera_order}")

    all_images = []

    for cam_idx in camera_order:
        cam_dir = os.path.join(input_dir, f'MyCam_{cam_idx:03d}')

        if not os.path.exists(cam_dir):
            print(f"  ❌ 카메라 {cam_idx} 없음: {cam_dir}")
            continue

        img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[:num_frames]
        images = []
        for f in img_files:
            img = cv2.imread(f)
            if img is not None:
                images.append(img)

        if crop_config:
            images = crop_edges(images, crop_config)

        if images:
            all_images.append(images)
            print(f"  ✅ 카메라 {cam_idx}: {len(images)}장")
        else:
            print(f"  ❌ 카메라 {cam_idx}: 유효 이미지 없음")

    print(f"  총 {len(all_images)}개 카메라\n")
    return all_images


class CuPyGPUStitcher:
    """CuPy GPU 가속 스티칭"""

    def __init__(self, scale=1.0, max_pano_w=12000, max_pano_h=4000, skip_reference_stitch=False):
        self.scale = scale
        self.max_pano_w = max_pano_w
        self.max_pano_h = max_pano_h
        self.skip_reference_stitch = skip_reference_stitch

        self.homographies = []
        self.pano_size = None   # (w, h)
        self.pano_offset = None # (min_x, min_y)
        self.calibrated = False

        self.sift = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        print(
            f"[GPU] CuPy 스티처 초기화 "
            f"(scale={scale}, max_pano={max_pano_w}x{max_pano_h}, skip_ref={skip_reference_stitch})"
        )

    def calibrate(self, images):
        """특징점 매칭으로 캘리브레이션"""
        print("\n" + "=" * 60)
        print(f"[캘리브레이션] 특징점 매칭 기반 (scale={self.scale})")
        print("=" * 60)

        if images is None or len(images) < 2:
            print("  ❌ 캘리브레이션 이미지 부족")
            self.calibrated = False
            return None

        # 리사이즈
        images_scaled = [
            cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            for img in images
        ]

        h, w = images_scaled[0].shape[:2]
        print(f"  입력: {len(images_scaled)}개 카메라 | {w}x{h}")

        # 특징점 검출
        print("\n[1/3] 특징점 검출 (SIFT)")
        keypoints_list = []
        descriptors_list = []

        for i, img in enumerate(images_scaled):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.sift.detectAndCompute(gray, None)
            keypoints_list.append(kp)
            descriptors_list.append(des)
            des_info = "None" if des is None else des.shape
            print(f"  카메라 {i+1}: {len(kp)}개 특징점 | des={des_info}")

            if des is None or len(kp) < 10:
                print(f"  ❌ 카메라 {i+1}: 특징점 부족")
                self.calibrated = False
                return None

        # 호모그래피 계산
        print("\n[2/3] 호모그래피 계산 (RANSAC)")
        self.homographies = [np.eye(3, dtype=np.float32)]  # 첫 이미지 기준

        for i in range(1, len(images_scaled)):
            prev_des = descriptors_list[i - 1]
            cur_des = descriptors_list[i]

            matches = self.matcher.knnMatch(prev_des, cur_des, k=2)

            good_matches = []
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 10:
                print(f"  ❌ 카메라 {i} ↔ {i+1}: 매칭 실패 ({len(good_matches)}개)")
                self.calibrated = False
                return None

            src_pts = np.float32(
                [keypoints_list[i - 1][m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints_list[i][m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            # dst -> src
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if H is None:
                print(f"  ❌ 카메라 {i} ↔ {i+1}: 호모그래피 실패")
                self.calibrated = False
                return None

            H_cumulative = self.homographies[-1] @ H
            self.homographies.append(H_cumulative.astype(np.float32))

            inliers = int(np.sum(mask)) if mask is not None else 0
            print(f"  ✅ 카메라 {i} ↔ {i+1}: {len(good_matches)}개 매칭, {inliers}개 inliers")

        # 파노라마 크기 계산
        print("\n[3/3] 파노라마 크기 계산")
        corners = []
        for H in self.homographies:
            corner = np.array(
                [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]],
                dtype=np.float32
            ).T
            transformed = H @ corner
            transformed /= transformed[2, :]
            corners.append(transformed[:2, :].T)

        all_corners = np.vstack(corners)
        min_x, min_y = all_corners.min(axis=0)
        max_x, max_y = all_corners.max(axis=0)

        self.pano_offset = (int(np.floor(min_x)), int(np.floor(min_y)))
        self.pano_size = (
            int(np.ceil(max_x - min_x)),
            int(np.ceil(max_y - min_y))
        )

        pano_w, pano_h = self.pano_size
        if pano_w <= 0 or pano_h <= 0:
            print(f"  ❌ 잘못된 파노라마 크기: {self.pano_size}")
            self.calibrated = False
            return None

        print(f"  파노라마: {pano_w}x{pano_h}")
        print(f"  오프셋: {self.pano_offset}")

        # 비정상 폭주 방지
        if pano_w > self.max_pano_w or pano_h > self.max_pano_h:
            print(f"  ❌ 파노라마 크기 과대: {pano_w}x{pano_h}")
            print(f"     제한값: {self.max_pano_w}x{self.max_pano_h}")
            print("     원인 후보: 호모그래피 누적 drift / 잘못된 매칭")
            self.calibrated = False
            return None

        self.calibrated = True

        # 기준 파노라마 생성 (선택)
        reference = None
        if self.skip_reference_stitch:
            print("\n[생성] 기준 파노라마 생성 생략 (--skip_reference_stitch)")
        else:
            print("\n[생성] 기준 파노라마")
            reference = self.stitch_gpu(images_scaled, already_scaled=True)
            if reference is None:
                print("  ❌ 기준 파노라마 생성 실패")
                self.calibrated = False
                return None

        print("\n✅ 캘리브레이션 완료!")
        print("=" * 60)
        return reference

    def stitch_gpu(self, images, already_scaled=False):
        """CuPy GPU 가속 스티칭 (einsum 제거 버전)"""
        if not self.calibrated:
            print("❌ 캘리브레이션 필요!")
            return None

        if images is None or len(images) == 0:
            return None

        # 리사이즈
        if already_scaled:
            images_scaled = images
        else:
            images_scaled = [
                cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
                for img in images
            ]

        if len(images_scaled) != len(self.homographies):
            print(f"❌ 이미지 수({len(images_scaled)})와 호모그래피 수({len(self.homographies)}) 불일치")
            return None

        pano_w, pano_h = self.pano_size
        offset_x, offset_y = self.pano_offset
        if pano_w <= 0 or pano_h <= 0:
            return None

        # 결과 캔버스
        try:
            result = cp.zeros((pano_h, pano_w, 3), dtype=cp.float32)
            weight_sum = cp.zeros((pano_h, pano_w), dtype=cp.float32)
        except cp.cuda.memory.OutOfMemoryError:
            print(f"❌ GPU 메모리 부족: pano={pano_w}x{pano_h}")
            return None

        h, w = images_scaled[0].shape[:2]

        # 파노라마 좌표 생성 (xx, yy만 사용)
        try:
            yy, xx = cp.meshgrid(
                cp.arange(pano_h, dtype=cp.float32),
                cp.arange(pano_w, dtype=cp.float32),
                indexing='ij'
            )
        except cp.cuda.memory.OutOfMemoryError:
            print(f"❌ meshgrid 생성 OOM: pano={pano_w}x{pano_h}")
            return None

        for img, H in zip(images_scaled, self.homographies):
            img_gpu = cp.asarray(img.astype(np.float32))

            H_offset = H.copy()
            H_offset[0, 2] -= offset_x
            H_offset[1, 2] -= offset_y
            H_offset_gpu = cp.asarray(H_offset, dtype=cp.float32)

            # 역변환
            H_inv = cp.linalg.inv(H_offset_gpu)

            # -------------------------------
            # cp.einsum 제거: 성분별 좌표 변환
            # -------------------------------
            h00, h01, h02 = H_inv[0, 0], H_inv[0, 1], H_inv[0, 2]
            h10, h11, h12 = H_inv[1, 0], H_inv[1, 1], H_inv[1, 2]
            h20, h21, h22 = H_inv[2, 0], H_inv[2, 1], H_inv[2, 2]

            den = h20 * xx + h21 * yy + h22
            den = cp.where(cp.abs(den) < 1e-6, 1e-6, den)

            src_x = (h00 * xx + h01 * yy + h02) / den
            src_y = (h10 * xx + h11 * yy + h12) / den

            # 유효 영역
            valid = (src_x >= 0) & (src_x < (w - 1)) & (src_y >= 0) & (src_y < (h - 1))

            # 바이리니어 보간
            x0f = cp.floor(src_x)
            y0f = cp.floor(src_y)
            x0 = cp.clip(x0f.astype(cp.int32), 0, w - 1)
            y0 = cp.clip(y0f.astype(cp.int32), 0, h - 1)
            x1 = cp.clip(x0 + 1, 0, w - 1)
            y1 = cp.clip(y0 + 1, 0, h - 1)

            wx = (src_x - x0f)[:, :, cp.newaxis]
            wy = (src_y - y0f)[:, :, cp.newaxis]

            warped = (
                img_gpu[y0, x0] * (1 - wx) * (1 - wy) +
                img_gpu[y0, x1] * wx * (1 - wy) +
                img_gpu[y1, x0] * (1 - wx) * wy +
                img_gpu[y1, x1] * wx * wy
            )

            # 가중치 (중앙 우선)
            center_x = w / 2.0
            center_y = h / 2.0
            dist_x = (src_x - center_x) / max(w / 2.0, 1.0)
            dist_y = (src_y - center_y) / max(h / 2.0, 1.0)
            dist = cp.sqrt(dist_x ** 2 + dist_y ** 2)
            weight = cp.exp(-(dist ** 2)) * valid.astype(cp.float32)

            result += warped * weight[:, :, cp.newaxis]
            weight_sum += weight

            # 임시 텐서 정리 힌트 (메모리 압박 완화)
            del img_gpu, H_offset_gpu, H_inv
            del den, src_x, src_y, valid, x0f, y0f, x0, y0, x1, y1, wx, wy, warped, weight
            # 너무 자주 free_all_blocks 하면 느려질 수 있어서 기본은 미사용
            # cp.get_default_memory_pool().free_all_blocks()

        result = result / cp.maximum(weight_sum, 1e-6)[:, :, cp.newaxis]
        out = cp.asnumpy(result)

        # cleanup
        del result, weight_sum, xx, yy
        return np.clip(out, 0, 255).astype(np.uint8)


class HybridCuPyStitcherV6:
    """하이브리드 CuPy GPU 스티칭"""

    def __init__(self, args):
        self.args = args
        self.stitcher = None
        self.receivers = []

        self.stitch_fps = 0.0
        self.stitch_count = 0
        self.stitch_time = time.time()

        self.frame_skip = 0
        self.skip_interval = 1

        # CSV 로깅
        self.csv_file = None
        self.csv_writer = None
        self.csv_rows_buffer = []

    def _init_csv_logger(self):
        if not self.args.save_csv:
            return

        csv_path = os.path.expanduser(self.args.save_csv)
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        self.csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)

        header = [
            "frame_idx", "timestamp",
            "receive_ms", "stitch_ms", "display_ms",
            "save_img_ms", "save_video_ms",
            "loop_ms", "loop_fps", "stitch_fps_est"
        ]
        for i in range(len(self.args.ports)):
            header.append(f"udp{i+1}_fps")

        self.csv_writer.writerow(header)
        self.csv_file.flush()
        print(f"[CSV] 성능 로그 저장: {csv_path}")

    def _flush_csv_buffer(self):
        if self.csv_file and self.csv_writer and self.csv_rows_buffer:
            self.csv_writer.writerows(self.csv_rows_buffer)
            self.csv_rows_buffer.clear()
            self.csv_file.flush()

    def _close_csv_logger(self):
        try:
            self._flush_csv_buffer()
        finally:
            if self.csv_file:
                self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            self.csv_rows_buffer = []

    def calibrate_from_folder(self):
        print("\n" + "=" * 60)
        print("[폴더] 캘리브레이션")
        print("=" * 60)

        # 크롭 설정
        if self.args.crop_edges > 0:
            crop_config = self.args.crop_edges
        elif any([self.args.crop_left, self.args.crop_right, self.args.crop_top, self.args.crop_bottom]):
            crop_config = (self.args.crop_left, self.args.crop_right, self.args.crop_top, self.args.crop_bottom)
        else:
            crop_config = None

        all_images = load_camera_images(
            os.path.expanduser(self.args.calibration_dir),
            num_frames=self.args.calibration_frames,
            crop_config=crop_config,
            camera_order=self.args.camera_order
        )

        if len(all_images) < 2:
            print("\n❌ 카메라 부족")
            return False

        min_len = min(len(cam_images) for cam_images in all_images)
        if min_len == 0:
            print("\n❌ 유효한 캘리브레이션 이미지가 없음")
            return False

        ref_idx = min(self.args.reference_frame, min_len - 1)
        ref_images = [cam_images[ref_idx] for cam_images in all_images]
        print(f"[기준] 프레임 {ref_idx + 1}/{min_len}\n")

        self.stitcher = CuPyGPUStitcher(
            scale=self.args.scale,
            max_pano_w=self.args.max_pano_w,
            max_pano_h=self.args.max_pano_h,
            skip_reference_stitch=self.args.skip_reference_stitch
        )

        reference = self.stitcher.calibrate(ref_images)

        # 성공 여부는 calibrated 플래그로 판정 (reference는 skip 모드에서 None 가능)
        if not self.stitcher.calibrated:
            return False

        if reference is not None and self.args.save_reference:
            ref_path = os.path.expanduser(self.args.save_reference)
            ref_dir = os.path.dirname(ref_path)
            if ref_dir:
                os.makedirs(ref_dir, exist_ok=True)
            ok = cv2.imwrite(ref_path, reference)
            if ok:
                print(f"\n[저장] {ref_path}")
            else:
                print(f"\n[경고] 기준 파노라마 저장 실패: {ref_path}")

        return True

    def init_receivers(self):
        print("\n" + "=" * 60)
        print("[UDP] 수신기 초기화")
        print("=" * 60)

        self.receivers = []
        for i, port in enumerate(self.args.ports):
            receiver = UDPReceiver(port, f"UDP {i+1}")
            receiver.start()
            self.receivers.append(receiver)

        print("\n  대기 중...")
        time.sleep(2)

    def get_8cam_images(self):
        """
        UDP 프레임을 받아 2x2 분할해서 8카메라 이미지 생성
        (포트 2개 x 각 포트 프레임 1개 x 2x2 분할 = 8개 이미지 가정)
        """
        frames_2cam = []
        for receiver in self.receivers:
            frame = receiver.get_frame(timeout=1.0)
            if frame is None:
                return None
            frames_2cam.append(frame)

        images_8cam = []
        for frame in frames_2cam:
            images_8cam.extend(split_2x2(frame))

        # 크롭 설정
        if self.args.crop_edges > 0:
            crop_config = self.args.crop_edges
        elif any([self.args.crop_left, self.args.crop_right, self.args.crop_top, self.args.crop_bottom]):
            crop_config = (self.args.crop_left, self.args.crop_right, self.args.crop_top, self.args.crop_bottom)
        else:
            crop_config = None

        if crop_config:
            images_8cam = crop_edges(images_8cam, crop_config)

        if self.args.camera_order:
            try:
                images_8cam = [images_8cam[i - 1] for i in self.args.camera_order]
            except Exception as e:
                print(f"[경고] camera_order 적용 실패: {e}")
                return None

        return images_8cam

    def run(self):
        if not self.calibrate_from_folder():
            print("\n❌ 캘리브레이션 실패!")
            return

        self.init_receivers()
        self._init_csv_logger()

        print("\n" + "=" * 60)
        print("[실시간] 스티칭 시작 (q=종료)")
        print("=" * 60 + "\n")

        frame_idx = 0
        success_count = 0
        video_writer = None

        if self.args.save_images:
            os.makedirs(os.path.expanduser(self.args.save_images), exist_ok=True)

        try:
            while True:
                loop_start = time.perf_counter()

                # 1) UDP 수신 시간
                t_recv = time.perf_counter()
                images_8cam = self.get_8cam_images()
                receive_ms = (time.perf_counter() - t_recv) * 1000.0

                if images_8cam is None:
                    continue

                # 프레임 스킵 (필요시 확장 가능)
                self.frame_skip += 1
                if self.frame_skip % self.skip_interval != 0:
                    continue

                # 2) 스티칭 시간
                t_stitch = time.perf_counter()
                pano = self.stitcher.stitch_gpu(images_8cam)
                cp.cuda.Stream.null.synchronize()
                stitch_ms = (time.perf_counter() - t_stitch) * 1000.0

                if pano is None:
                    continue

                success_count += 1

                # 표시용 FPS 계산(기존 스타일 유지)
                self.stitch_count += 1
                now = time.time()
                if now - self.stitch_time >= 1.0:
                    self.stitch_fps = self.stitch_count / (now - self.stitch_time)
                    self.stitch_count = 0
                    self.stitch_time = now

                # 오버레이
                info = f"FPS: {self.stitch_fps:.1f} | Frame: {frame_idx} | CuPy GPU v6"
                for i, r in enumerate(self.receivers):
                    info += f" | UDP{i+1}: {r.fps:.1f}"

                cv2.putText(
                    pano, info, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )

                # 비디오 저장기 초기화
                if video_writer is None and self.args.save_video:
                    out_video = os.path.expanduser(self.args.save_video)
                    out_dir = os.path.dirname(out_video)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        out_video, fourcc, self.args.fps,
                        (pano.shape[1], pano.shape[0])
                    )
                    if not video_writer.isOpened():
                        print(f"[경고] 비디오 저장기 열기 실패: {out_video}")
                        video_writer = None
                    else:
                        print(f"[비디오] {out_video}")

                # 3) 표시 시간
                t_display = time.perf_counter()
                display_h = 600
                display_w = int(display_h * pano.shape[1] / max(pano.shape[0], 1))
                display = cv2.resize(pano, (display_w, display_h))
                cv2.imshow('CuPy GPU Panorama v6', display)
                display_ms = (time.perf_counter() - t_display) * 1000.0

                # 4) 저장 시간
                save_video_ms = 0.0
                save_img_ms = 0.0

                if video_writer is not None:
                    t_sv = time.perf_counter()
                    video_writer.write(pano)
                    save_video_ms = (time.perf_counter() - t_sv) * 1000.0

                if self.args.save_images:
                    t_si = time.perf_counter()
                    img_dir = os.path.expanduser(self.args.save_images)
                    img_path = os.path.join(img_dir, f"pano_{frame_idx:04d}.jpg")
                    cv2.imwrite(img_path, pano)
                    save_img_ms = (time.perf_counter() - t_si) * 1000.0

                # 5) 루프 전체 시간 / FPS
                loop_ms = (time.perf_counter() - loop_start) * 1000.0
                loop_fps = (1000.0 / loop_ms) if loop_ms > 0 else 0.0
                stitch_fps_est = (1000.0 / stitch_ms) if stitch_ms > 0 else 0.0

                # 6) CSV 기록
                if self.csv_writer:
                    row = [
                        frame_idx,
                        datetime.now().isoformat(timespec="milliseconds"),
                        round(receive_ms, 3),
                        round(stitch_ms, 3),
                        round(display_ms, 3),
                        round(save_img_ms, 3),
                        round(save_video_ms, 3),
                        round(loop_ms, 3),
                        round(loop_fps, 3),
                        round(stitch_fps_est, 3),
                    ]
                    row.extend([round(r.fps, 3) for r in self.receivers])

                    self.csv_rows_buffer.append(row)

                    if len(self.csv_rows_buffer) >= max(1, self.args.csv_flush_interval):
                        self._flush_csv_buffer()

                if (frame_idx + 1) % 10 == 0:
                    print(
                        f"  프레임 {frame_idx+1} | "
                        f"loop={loop_fps:.2f} FPS | stitch={stitch_fps_est:.2f} FPS | "
                        f"recv={receive_ms:.1f}ms stitch={stitch_ms:.1f}ms "
                        f"display={display_ms:.1f}ms save_img={save_img_ms:.1f}ms save_vid={save_video_ms:.1f}ms"
                    )

                frame_idx += 1

                if cv2.waitKey(1) == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n\n중단됨")

        finally:
            for r in self.receivers:
                r.stop()

            if video_writer is not None:
                video_writer.release()

            cv2.destroyAllWindows()
            self._close_csv_logger()

            print("\n" + "=" * 60)
            print("[통계]")
            print("=" * 60)
            print(f"성공: {success_count} 프레임")
            print(f"표시용 FPS(최근): {self.stitch_fps:.2f}")

            if self.args.save_video:
                print(f"✅ 비디오: {os.path.expanduser(self.args.save_video)}")
            if self.args.save_images:
                print(f"✅ 이미지: {os.path.expanduser(self.args.save_images)} ({success_count}장)")
            if self.args.save_csv:
                print(f"✅ CSV: {os.path.expanduser(self.args.save_csv)}")


def main():
    parser = argparse.ArgumentParser(description="CuPy GPU 파노라마 스티칭 v6 + CSV 성능 로깅 (안정화 버전)")

    # 캘리브레이션
    parser.add_argument("--calibration_dir", type=str, required=True)
    parser.add_argument("--calibration_frames", type=int, default=10)
    parser.add_argument("--reference_frame", type=int, default=7)

    # UDP
    parser.add_argument("--ports", type=int, nargs="+", default=[5001, 5002])

    # 카메라 순서 (1-based index)
    parser.add_argument("--camera_order", type=int, nargs="+", default=None)

    # 크롭
    parser.add_argument("--crop_edges", type=int, default=0)
    parser.add_argument("--crop_left", type=int, default=0)
    parser.add_argument("--crop_right", type=int, default=0)
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)

    # 성능/스케일
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--target_fps", type=int, default=10)

    # 저장
    parser.add_argument("--save_images", type=str, default=None)
    parser.add_argument("--save_video", type=str, default=None)
    parser.add_argument("--save_reference", type=str, default="reference_cupy_gpu_v6.jpg")
    parser.add_argument("--fps", type=int, default=10)

    # CSV 성능 로그
    parser.add_argument("--save_csv", type=str, default="perf_log.csv",
                        help="성능 로그 CSV 저장 경로")
    parser.add_argument("--csv_flush_interval", type=int, default=10,
                        help="N프레임마다 CSV flush")

    # 안정화 옵션
    parser.add_argument("--skip_reference_stitch", action="store_true",
                        help="캘리브레이션 후 기준 파노라마 생성 생략 (OOM/CUBLAS 에러 방지)")
    parser.add_argument("--max_pano_w", type=int, default=12000,
                        help="파노라마 최대 너비 제한")
    parser.add_argument("--max_pano_h", type=int, default=4000,
                        help="파노라마 최대 높이 제한")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CuPy GPU 파노라마 스티칭 v6 + CSV 로깅 (안정화)")
    print("=" * 60)
    print(f"캘리브레이션: {args.calibration_dir}")
    print(f"기준 프레임: {args.reference_frame}")
    print(f"UDP 포트: {args.ports}")
    print(f"스케일: {args.scale * 100:.0f}%")
    print(f"목표 FPS: {args.target_fps}")
    print(f"CSV 로그: {args.save_csv}")
    print(f"기준 파노라마 생략: {args.skip_reference_stitch}")
    print(f"파노라마 제한: {args.max_pano_w}x{args.max_pano_h}")
    if args.camera_order:
        print(f"카메라 순서: {args.camera_order}")
    print("=" * 60)

    app = HybridCuPyStitcherV6(args)
    app.run()


if __name__ == "__main__":
    main()