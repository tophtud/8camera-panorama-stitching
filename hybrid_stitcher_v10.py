#!/usr/bin/env python3
"""
하이브리드 파노라마 스티칭 v10 (OpenCV Stitcher)
- 폴더 이미지로 캘리브레이션
- UDP 실시간 스티칭 (매 프레임 stitch 호출)
- 안정적이지만 느림 (1-2 FPS)
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
except:
    print("[⚠️ OpenCV CUDA] CUDA 지원 없음")


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


class OpenCVStitcher:
    """OpenCV Stitcher 래퍼"""
    
    def __init__(self, scale=1.0):
        self.scale = scale
        self.stitcher = None
        self.calibrated = False
        
        print(f"[스티처] OpenCV Stitcher (scale={scale})")
    
    def calibrate(self, images):
        """캘리브레이션"""
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
        print("\n[1/2] OpenCV Stitcher 생성")
        self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        self.stitcher.setRegistrationResol(0.6)
        self.stitcher.setSeamEstimationResol(0.1)
        self.stitcher.setCompositingResol(-1)
        self.stitcher.setPanoConfidenceThresh(0.5)
        
        print("  등록 해상도: 0.6 MP")
        print("  Seam 해상도: 0.1 MP")
        print("  합성 해상도: 원본")
        print("  신뢰도 임계값: 0.5")
        
        # 변환 추정
        print("\n[2/2] 변환 추정")
        status = self.stitcher.estimateTransform(images_scaled)
        
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 변환 추정 실패: {status}")
            return None
        
        print("  ✅ 변환 추정 성공!")
        
        # 기준 파노라마 생성
        print("\n[생성] 기준 파노라마")
        status, reference = self.stitcher.composePanorama()
        
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 파노라마 생성 실패: {status}")
            return None
        
        print(f"  ✅ 파노라마: {reference.shape[1]}x{reference.shape[0]}")
        
        self.calibrated = True
        
        print("\n✅ 캘리브레이션 완료!")
        print("=" * 60)
        
        return reference
    
    def stitch(self, images):
        """스티칭 (매 프레임)"""
        if not self.calibrated:
            print("❌ 캘리브레이션 필요!")
            return None
        
        # 리사이즈
        images_scaled = [
            cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            for img in images
        ]
        
        # 스티칭
        status, panorama = self.stitcher.stitch(images_scaled)
        
        if status != cv2.Stitcher_OK:
            return None
        
        return panorama


class HybridStitcherV10:
    """하이브리드 스티칭 v10"""
    
    def __init__(self, args):
        self.args = args
        self.stitcher = None
        self.receivers = []
        
        self.stitch_fps = 0
        self.stitch_count = 0
        self.stitch_time = time.time()
        
        self.frame_skip = 0
        self.skip_interval = max(1, 30 // args.target_fps) if args.target_fps > 0 else 1
    
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
        self.stitcher = OpenCVStitcher(scale=self.args.scale)
        
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
        print("[실시간] OpenCV Stitcher (q=종료)")
        print("=" * 60)
        print(f"프레임 스킵: {self.skip_interval-1} (목표 FPS: {self.args.target_fps})\n")
        
        frame_idx = 0
        success_count = 0
        video_writer = None
        
        if self.args.save_images:
            os.makedirs(self.args.save_images, exist_ok=True)
        
        if self.args.save_video:
            os.makedirs(os.path.dirname(self.args.save_video) if os.path.dirname(self.args.save_video) else '.', exist_ok=True)
        
        try:
            while True:
                # 수신
                images_8cam = self.get_8cam_images()
                if images_8cam is None:
                    continue
                
                # 프레임 스킵
                self.frame_skip += 1
                if self.frame_skip < self.skip_interval:
                    continue
                self.frame_skip = 0
                
                # 스티칭
                start_time = time.time()
                pano = self.stitcher.stitch(images_8cam)
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
                info = f"FPS: {self.stitch_fps:.1f} | Frame: {frame_idx} | OpenCV v10"
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
                cv2.imshow('Panorama v10 (OpenCV)', display)
                
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
    parser = argparse.ArgumentParser(description="하이브리드 파노라마 스티칭 v10 (OpenCV Stitcher)")
    
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
    parser.add_argument("--target_fps", type=int, default=5)
    parser.add_argument("--save_images", type=str, default=None)
    parser.add_argument("--save_video", type=str, default=None)
    parser.add_argument("--save_reference", type=str, default="reference_v10.jpg")
    parser.add_argument("--fps", type=int, default=5)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("하이브리드 파노라마 스티칭 v10 (OpenCV Stitcher)")
    print("=" * 60)
    print(f"캘리브레이션: {args.calibration_dir}")
    print(f"기준 프레임: {args.reference_frame}")
    print(f"UDP 포트: {args.ports}")
    print(f"스케일: {args.scale * 100:.0f}%")
    print(f"목표 FPS: {args.target_fps}")
    if args.camera_order:
        print(f"카메라 순서: {args.camera_order}")
    print("=" * 60)
    
    stitcher = HybridStitcherV10(args)
    stitcher.run()


if __name__ == "__main__":
    main()
