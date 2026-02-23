#!/usr/bin/env python3
"""
하이브리드 파노라마 스티칭 v9 (OpenCV Stitcher + CuPy GPU)
- OpenCV Stitcher로 안정적인 호모그래피 추출
- CuPy GPU 가속 실시간 워핑/블렌딩
- 10+ FPS 목표
"""

import os
import glob
import argparse
import time
import numpy as np
import cv2
import threading
from queue import Queue, Empty

# OpenCV CUDA 확인
try:
    gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
    if gpu_count > 0:
        print(f"[✅ OpenCV CUDA] {gpu_count}개 GPU 활성화")
    else:
        print("[❌ OpenCV CUDA] CUDA GPU 없음")
except:
    print("[❌ OpenCV CUDA] CUDA 지원 없음")

# CuPy 확인
try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    if CUPY_AVAILABLE:
        print(f"[✅ CuPy] {cp.__version__} | {cp.cuda.runtime.getDeviceCount()}개 GPU 활성화")
        # GPU 워밍업
        _ = cp.array([1])
        cp.cuda.Stream.null.synchronize()
    else:
        print("[⚠️ CuPy] GPU 없음 - CPU 모드")
        CUPY_AVAILABLE = False
except ImportError:
    print("[⚠️ CuPy] 설치 안 됨 - CPU 모드")
    print("   설치: sudo pip3 install cupy-cuda11x")
    CUPY_AVAILABLE = False


class UDPReceiver:
    """UDP 스트림 수신"""
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
        return [img[c:-c, c:-c] if c > 0 else img for img in images]
    
    cl, cr, ct, cb = crop_config
    return [img[ct:-cb if cb > 0 else None, cl:-cr if cr > 0 else None] for img in images]


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
            print(f"  ❌ 카메라 {cam_idx} 없음")
            continue
        
        img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[:num_frames]
        images = [cv2.imread(f) for f in img_files if cv2.imread(f) is not None]
        
        if images:
            all_images.append(images)
            print(f"  ✅ 카메라 {cam_idx}: {len(images)}장")
    
    print(f"  총 {len(all_images)}개 카메라\n")
    return all_images


class HybridGPUStitcher:
    """OpenCV Stitcher + CuPy GPU 하이브리드"""
    
    def __init__(self, scale=1.0):
        self.scale = scale
        self.homographies = []
        self.pano_size = None
        self.pano_offset = None
        self.calibrated = False
        self.use_gpu = CUPY_AVAILABLE
        
        print(f"[스티처] 초기화 (scale={scale}, GPU={'ON' if self.use_gpu else 'OFF'})")
    
    def calibrate(self, images):
        """OpenCV Stitcher로 캘리브레이션"""
        print("\n" + "=" * 60)
        print(f"[캘리브레이션] OpenCV Stitcher (scale={self.scale})")
        print("=" * 60)
        
        # 리사이즈
        images_scaled = [
            cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            for img in images
        ]
        
        h, w = images_scaled[0].shape[:2]
        print(f"  입력: {len(images_scaled)}개 카메라 | {w}x{h}")
        
        # OpenCV Stitcher
        print("\n[1/3] OpenCV Stitcher 변환 추정")
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        stitcher.setRegistrationResol(0.6)
        stitcher.setSeamEstimationResol(0.1)
        stitcher.setCompositingResol(-1)
        stitcher.setPanoConfidenceThresh(0.5)
        
        status = stitcher.estimateTransform(images_scaled)
        
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 변환 추정 실패: {status}")
            return None
        
        print("  ✅ 변환 추정 성공!")
        
        # 기준 파노라마 생성 (크기 확인용)
        print("\n[2/3] 기준 파노라마 생성")
        status, reference = stitcher.composePanorama()
        
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 파노라마 생성 실패: {status}")
            return None
        
        print(f"  ✅ 파노라마: {reference.shape[1]}x{reference.shape[0]}")
        
        # 호모그래피 추출
        print("\n[3/3] 호모그래피 추출")
        cameras = stitcher.cameras()
        
        if not cameras:
            print("  ❌ 카메라 정보 없음!")
            return None
        
        print(f"  ✅ {len(cameras)}개 카메라")
        
        # 각 카메라의 호모그래피 계산
        self.homographies = []
        corners = []
        
        for i, cam in enumerate(cameras):
            # 카메라 내부 파라미터
            K = cam.K().astype(np.float32)
            R = cam.R.astype(np.float32)
            
            # 호모그래피 = K * R
            H = K @ R
            self.homographies.append(H)
            
            # 코너 계산
            corner = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32).T
            transformed = H @ corner
            transformed /= transformed[2, :]
            corners.append(transformed[:2, :].T)
        
        # 파노라마 크기 계산
        all_corners = np.vstack(corners)
        min_x, min_y = all_corners.min(axis=0)
        max_x, max_y = all_corners.max(axis=0)
        
        self.pano_offset = (int(min_x), int(min_y))
        self.pano_size = (int(max_x - min_x), int(max_y - min_y))
        
        print(f"  파노라마: {self.pano_size[0]}x{self.pano_size[1]}")
        print(f"  오프셋: {self.pano_offset}")
        
        # 평행이동 보정
        for i in range(len(self.homographies)):
            T = np.array([
                [1, 0, -self.pano_offset[0]],
                [0, 1, -self.pano_offset[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            self.homographies[i] = T @ self.homographies[i]
        
        self.calibrated = True
        
        print("\n✅ 캘리브레이션 완료!")
        print("=" * 60)
        
        return reference
    
    def stitch_gpu(self, images):
        """CuPy GPU 가속 스티칭"""
        if not self.calibrated:
            print("❌ 캘리브레이션 필요!")
            return None
        
        # 리사이즈
        images_scaled = [
            cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            for img in images
        ]
        
        if not self.use_gpu:
            return self._stitch_cpu(images_scaled)
        
        pano_w, pano_h = self.pano_size
        
        # GPU 배열
        result = cp.zeros((pano_h, pano_w, 3), dtype=cp.float32)
        weight_sum = cp.zeros((pano_h, pano_w), dtype=cp.float32)
        
        h, w = images_scaled[0].shape[:2]
        
        for img, H in zip(images_scaled, self.homographies):
            # 이미지를 GPU로
            img_gpu = cp.asarray(img.astype(np.float32))
            H_gpu = cp.asarray(H)
            
            # 워핑 (역변환)
            H_inv = cp.linalg.inv(H_gpu)
            
            # 파노라마 좌표 생성
            yy, xx = cp.meshgrid(cp.arange(pano_h), cp.arange(pano_w), indexing='ij')
            coords = cp.stack([xx, yy, cp.ones_like(xx)], axis=-1).astype(cp.float32)
            
            # 역변환
            src_coords = cp.einsum('ij,hwj->hwi', H_inv, coords)
            src_coords = src_coords[:, :, :2] / src_coords[:, :, 2:3]
            
            src_x = src_coords[:, :, 0]
            src_y = src_coords[:, :, 1]
            
            # 유효 영역 마스크
            valid = (src_x >= 0) & (src_x < w - 1) & (src_y >= 0) & (src_y < h - 1)
            
            # 바이리니어 보간
            x0 = cp.floor(src_x).astype(cp.int32)
            y0 = cp.floor(src_y).astype(cp.int32)
            x1 = cp.clip(x0 + 1, 0, w - 1)
            y1 = cp.clip(y0 + 1, 0, h - 1)
            x0 = cp.clip(x0, 0, w - 1)
            y0 = cp.clip(y0, 0, h - 1)
            
            wx = (src_x - cp.floor(src_x))[:, :, cp.newaxis]
            wy = (src_y - cp.floor(src_y))[:, :, cp.newaxis]
            
            warped = (
                img_gpu[y0, x0] * (1 - wx) * (1 - wy) +
                img_gpu[y0, x1] * wx * (1 - wy) +
                img_gpu[y1, x0] * (1 - wx) * wy +
                img_gpu[y1, x1] * wx * wy
            )
            
            # 가중치 (가우시안)
            center_x = w / 2
            center_y = h / 2
            dist_x = (src_x - center_x) / (w / 2)
            dist_y = (src_y - center_y) / (h / 2)
            dist = cp.sqrt(dist_x ** 2 + dist_y ** 2)
            weight = cp.exp(-dist ** 2) * valid.astype(cp.float32)
            
            # 누적
            result += warped * weight[:, :, cp.newaxis]
            weight_sum += weight
        
        # 정규화
        result = result / cp.maximum(weight_sum, 1e-6)[:, :, cp.newaxis]
        
        return np.clip(cp.asnumpy(result), 0, 255).astype(np.uint8)
    
    def _stitch_cpu(self, images_scaled):
        """CPU 스티칭 (폴백)"""
        pano_w, pano_h = self.pano_size
        
        result = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        weight_sum = np.zeros((pano_h, pano_w), dtype=np.float32)
        
        h, w = images_scaled[0].shape[:2]
        
        for img, H in zip(images_scaled, self.homographies):
            # 워핑
            warped = cv2.warpPerspective(img, H, (pano_w, pano_h))
            
            # 마스크
            mask = cv2.warpPerspective(
                np.ones((h, w), dtype=np.uint8) * 255,
                H, (pano_w, pano_h)
            ) / 255.0
            
            # 누적
            result += warped.astype(np.float32) * mask[:, :, np.newaxis]
            weight_sum += mask
        
        # 정규화
        valid = weight_sum > 0
        result[valid] /= weight_sum[valid, np.newaxis]
        
        return np.clip(result, 0, 255).astype(np.uint8)


class HybridStitcherV9:
    """하이브리드 스티칭 v9"""
    
    def __init__(self, args):
        self.args = args
        self.stitcher = None
        self.receivers = []
        
        self.stitch_fps = 0
        self.stitch_count = 0
        self.stitch_time = time.time()
        
        self.frame_skip = 0
        self.skip_interval = 1
    
    def calibrate_from_folder(self):
        """폴더 이미지로 캘리브레이션"""
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
        
        # 이미지 로드
        all_images = load_camera_images(
            os.path.expanduser(self.args.calibration_dir),
            num_frames=self.args.calibration_frames,
            crop_config=crop_config,
            camera_order=self.args.camera_order
        )
        
        if len(all_images) < 2:
            print("\n❌ 카메라 부족")
            return False
        
        # 기준 프레임
        ref_idx = min(self.args.reference_frame - 1, len(all_images[0]) - 1)
        ref_images = [cam_images[ref_idx] for cam_images in all_images]
        
        print(f"[기준] 프레임 {ref_idx+1}/{len(all_images[0])}\n")
        
        # 스티처 초기화
        self.stitcher = HybridGPUStitcher(scale=self.args.scale)
        
        # 캘리브레이션
        reference = self.stitcher.calibrate(ref_images)
        
        if reference is None:
            return False
        
        # 저장
        if self.args.save_reference:
            cv2.imwrite(self.args.save_reference, reference)
            print(f"\n[저장] {self.args.save_reference}")
        
        return True
    
    def init_receivers(self):
        """UDP 수신기 초기화"""
        print("\n" + "=" * 60)
        print("[UDP] 수신기 초기화")
        print("=" * 60)
        
        for i, port in enumerate(self.args.ports):
            receiver = UDPReceiver(port, f"UDP {i+1}")
            receiver.start()
            self.receivers.append(receiver)
        
        print("\n  대기 중...")
        time.sleep(2)
    
    def get_8cam_images(self):
        """UDP에서 8개 카메라 이미지"""
        frames_2cam = []
        for receiver in self.receivers:
            frame = receiver.get_frame(timeout=1.0)
            if frame is None:
                return None
            frames_2cam.append(frame)
        
        # 2x2 분할
        images_8cam = []
        for frame in frames_2cam:
            images_8cam.extend(split_2x2(frame))
        
        # 크롭
        if self.args.crop_edges > 0:
            crop_config = self.args.crop_edges
        elif any([self.args.crop_left, self.args.crop_right, self.args.crop_top, self.args.crop_bottom]):
            crop_config = (self.args.crop_left, self.args.crop_right, self.args.crop_top, self.args.crop_bottom)
        else:
            crop_config = None
        
        if crop_config:
            images_8cam = crop_edges(images_8cam, crop_config)
        
        # 순서
        if self.args.camera_order:
            images_8cam = [images_8cam[i-1] for i in self.args.camera_order]
        
        return images_8cam
    
    def run(self):
        """실행"""
        # 캘리브레이션
        if not self.calibrate_from_folder():
            print("\n❌ 캘리브레이션 실패!")
            return
        
        # UDP 초기화
        self.init_receivers()
        
        # 실시간 스티칭
        print("\n" + "=" * 60)
        print("[실시간] GPU 스티칭 시작 (q=종료)")
        print("=" * 60 + "\n")
        
        frame_idx = 0
        success_count = 0
        video_writer = None
        
        if self.args.save_images:
            os.makedirs(self.args.save_images, exist_ok=True)
        
        if self.args.save_video:
            os.makedirs(os.path.dirname(self.args.save_video), exist_ok=True)
        
        try:
            while True:
                # 수신
                images_8cam = self.get_8cam_images()
                if images_8cam is None:
                    continue
                
                # 프레임 스킵
                self.frame_skip += 1
                if self.frame_skip % self.skip_interval != 0:
                    continue
                
                # 스티칭
                start_time = time.time()
                pano = self.stitcher.stitch_gpu(images_8cam)
                if CUPY_AVAILABLE:
                    cp.cuda.Stream.null.synchronize()
                stitch_time = time.time() - start_time
                
                if pano is None:
                    continue
                
                success_count += 1
                
                # FPS
                self.stitch_count += 1
                current_time = time.time()
                if current_time - self.stitch_time >= 1.0:
                    self.stitch_fps = self.stitch_count / (current_time - self.stitch_time)
                    self.stitch_count = 0
                    self.stitch_time = current_time
                
                # 정보
                mode = "GPU" if CUPY_AVAILABLE else "CPU"
                info = f"FPS: {self.stitch_fps:.1f} | Frame: {frame_idx} | {mode} v9"
                for i, r in enumerate(self.receivers):
                    info += f" | UDP{i+1}: {r.fps:.1f}"
                
                cv2.putText(pano, info, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # 비디오 저장
                if video_writer is None and self.args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        self.args.save_video, fourcc, self.args.fps,
                        (pano.shape[1], pano.shape[0])
                    )
                    print(f"[비디오] {self.args.save_video}")
                
                # 화면
                display_h = 600
                display_w = int(display_h * pano.shape[1] / pano.shape[0])
                display = cv2.resize(pano, (display_w, display_h))
                cv2.imshow(f'Panorama v9 ({mode})', display)
                
                if video_writer:
                    video_writer.write(pano)
                
                if self.args.save_images:
                    cv2.imwrite(os.path.join(self.args.save_images, f'pano_{frame_idx:04d}.jpg'), pano)
                
                if (frame_idx + 1) % 10 == 0:
                    print(f"  프레임 {frame_idx+1} | {self.stitch_fps:.1f} FPS")
                
                frame_idx += 1
                
                if cv2.waitKey(1) == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n\n중단됨")
        
        finally:
            for r in self.receivers:
                r.stop()
            
            if video_writer:
                video_writer.release()
            
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print("[통계]")
            print("=" * 60)
            print(f"성공: {success_count} 프레임")
            print(f"평균 FPS: {self.stitch_fps:.2f}")
            
            if self.args.save_video:
                print(f"\n✅ 비디오: {self.args.save_video}")
            if self.args.save_images:
                print(f"✅ 이미지: {self.args.save_images} ({success_count}장)")


def main():
    parser = argparse.ArgumentParser(description="하이브리드 파노라마 스티칭 v9 (OpenCV Stitcher + CuPy GPU)")
    
    parser.add_argument("--calibration_dir", type=str, required=True)
    parser.add_argument("--calibration_frames", type=int, default=10)
    parser.add_argument("--reference_frame", type=int, default=7)
    parser.add_argument("--ports", type=int, nargs="+", default=[5001, 5002])
    parser.add_argument("--camera_order", type=int, nargs="+", default=None)
    parser.add_argument("--crop_edges", type=int, default=0)
    parser.add_argument("--crop_left", type=int, default=0)
    parser.add_argument("--crop_right", type=int, default=0)
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--target_fps", type=int, default=10)
    parser.add_argument("--save_images", type=str, default=None)
    parser.add_argument("--save_video", type=str, default=None)
    parser.add_argument("--save_reference", type=str, default="reference_v9.jpg")
    parser.add_argument("--fps", type=int, default=10)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("하이브리드 파노라마 스티칭 v9 (OpenCV + CuPy GPU)")
    print("=" * 60)
    print(f"캘리브레이션: {args.calibration_dir}")
    print(f"기준 프레임: {args.reference_frame}")
    print(f"UDP 포트: {args.ports}")
    print(f"스케일: {args.scale * 100:.0f}%")
    print(f"목표 FPS: {args.target_fps}")
    if args.camera_order:
        print(f"카메라 순서: {args.camera_order}")
    print("=" * 60)
    
    stitcher = HybridStitcherV9(args)
    stitcher.run()


if __name__ == "__main__":
    main()
