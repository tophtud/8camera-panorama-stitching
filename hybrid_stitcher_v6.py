#!/usr/bin/env python3
"""
하이브리드 파노라마 스티칭 v6
- 폴더 이미지로 캘리브레이션 (한 번만)
- UDP 스트림으로 실시간 스티칭
"""

import os
import glob
import argparse
import time
import numpy as np
import cv2
import threading
from queue import Queue, Empty


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
        crop_left = crop_right = crop_top = crop_bottom = crop_config
    else:
        crop_left, crop_right, crop_top, crop_bottom = crop_config
    
    cropped = []
    for img in images:
        if crop_left > 0:
            img[:, :crop_left] = 0
        if crop_right > 0:
            img[:, -crop_right:] = 0
        if crop_top > 0:
            img[:crop_top, :] = 0
        if crop_bottom > 0:
            img[-crop_bottom:, :] = 0
        
        cropped.append(img)
    
    return cropped


def load_camera_images(input_dir, num_cameras=8, num_frames=10, crop_config=None, camera_order=None):
    """폴더에서 이미지 로드"""
    print("\n폴더 이미지 로드 중...")
    
    if camera_order is None:
        camera_order = list(range(1, num_cameras + 1))
    
    print(f"  카메라 순서: {camera_order}")
    
    all_images = []
    
    for cam_idx in camera_order:
        cam_dir = os.path.join(input_dir, f'MyCam_{cam_idx:03d}')
        
        if not os.path.exists(cam_dir):
            print(f"  ⚠️ 카메라 {cam_idx} 디렉토리 없음")
            continue
        
        img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
        
        if not img_files:
            print(f"  ⚠️ 카메라 {cam_idx} 이미지 없음")
            continue
        
        images = []
        for img_file in img_files[:num_frames]:
            img = cv2.imread(img_file)
            if img is not None:
                images.append(img)
        
        if images:
            all_images.append(images)
            print(f"  ✅ 카메라 {cam_idx}: {len(images)}장")
    
    print(f"\n✅ 총 {len(all_images)}개 카메라")
    
    return all_images


class HybridStitcherV6:
    """하이브리드 스티칭 (폴더 캘리브레이션 + UDP 실시간)"""
    
    def __init__(self, args):
        self.args = args
        self.stitcher = None
        self.is_calibrated = False
        self.receivers = []
        
        # 통계
        self.stitch_fps = 0
        self.stitch_count = 0
        self.stitch_time = time.time()
        
        # 프레임 스킵
        self.frame_skip = 0
        self.skip_interval = 3 if args.target_fps == 10 else 2
        
    def calibrate_from_folder(self):
        """폴더 이미지로 캘리브레이션"""
        print("=" * 60)
        print("폴더 이미지로 캘리브레이션")
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
            print("\n❌ 카메라가 부족합니다")
            return False
        
        # 기준 프레임 추출
        ref_idx = min(self.args.reference_frame, len(all_images[0]) - 1)
        ref_images = [cam_images[ref_idx] for cam_images in all_images]
        
        print(f"\n기준 프레임: {ref_idx+1}/{len(all_images[0])}")
        
        # 리사이즈
        images_scaled = [
            cv2.resize(img, None, fx=self.args.scale, fy=self.args.scale, interpolation=cv2.INTER_AREA)
            for img in ref_images
        ]
        
        h, w = images_scaled[0].shape[:2]
        print(f"입력: {len(images_scaled)}개 카메라")
        print(f"크기: {w} x {h}")
        
        # OpenCV Stitcher 생성
        print("\nOpenCV Stitcher 생성 중...")
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        
        # 고급 설정
        try:
            self.stitcher.setRegistrationResol(0.6)
            self.stitcher.setSeamEstimationResol(0.1)
            self.stitcher.setCompositingResol(-1)
            self.stitcher.setPanoConfidenceThresh(0.5)
            print("  등록 해상도: 0.6 MP")
            print("  Seam 해상도: 0.1 MP")
            print("  합성 해상도: 원본")
            print("  신뢰도 임계값: 0.5")
        except Exception as e:
            print(f"  설정 적용 실패: {e}")
        
        # 캘리브레이션
        print("\n변환 추정 중...")
        status = self.stitcher.estimateTransform(images_scaled)
        
        if status != cv2.Stitcher_OK:
            print(f"❌ 캘리브레이션 실패: {status}")
            return False
        
        print("✅ 변환 추정 성공!")
        
        # 기준 파노라마 생성
        print("\n기준 파노라마 생성 중...")
        status2, reference = self.stitcher.composePanorama()
        
        if status2 != cv2.Stitcher_OK:
            print(f"❌ 파노라마 생성 실패: {status2}")
            return False
        
        print(f"✅ 기준 파노라마 생성 성공!")
        print(f"크기: {reference.shape[1]} x {reference.shape[0]}")
        
        # 기준 파노라마 저장
        if self.args.save_reference:
            cv2.imwrite(self.args.save_reference, reference)
            print(f"\n✅ 기준 파노라마 저장: {self.args.save_reference}")
        
        self.is_calibrated = True
        
        print("\n✅ 캘리브레이션 완료!")
        print("=" * 60)
        
        return True
    
    def init_receivers(self):
        """UDP 수신기 초기화"""
        print("\n" + "=" * 60)
        print("UDP 수신기 초기화")
        print("=" * 60)
        
        for i, port in enumerate(self.args.ports):
            receiver = UDPReceiver(port, f"카메라 {i+1}")
            receiver.start()
            self.receivers.append(receiver)
            
        print("  프레임 수신 대기 중...")
        time.sleep(2)
    
    def get_8cam_images(self):
        """UDP에서 8개 카메라 이미지 추출"""
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
            
        # 가장자리 크롭
        if self.args.crop_edges > 0:
            crop_config = self.args.crop_edges
        elif any([self.args.crop_left, self.args.crop_right, self.args.crop_top, self.args.crop_bottom]):
            crop_config = (self.args.crop_left, self.args.crop_right, self.args.crop_top, self.args.crop_bottom)
        else:
            crop_config = None
            
        if crop_config:
            images_8cam = crop_edges(images_8cam, crop_config)
        
        # 카메라 순서 재배치
        if self.args.camera_order:
            images_8cam = [images_8cam[i-1] for i in self.args.camera_order]
            
        return images_8cam
    
    def stitch(self, images):
        """이미지 스티칭 (캘리브레이션된 Stitcher 사용)"""
        if not self.is_calibrated:
            return None
        
        # 리사이즈
        images_scaled = [
            cv2.resize(img, None, fx=self.args.scale, fy=self.args.scale, interpolation=cv2.INTER_AREA)
            for img in images
        ]
        
        # 스티칭 (캘리브레이션 재사용)
        status, panorama = self.stitcher.composePanorama(images_scaled)
        
        if status != cv2.Stitcher_OK:
            return None
        
        return panorama
    
    def run(self):
        """실행"""
        # 1단계: 폴더 이미지로 캘리브레이션
        if not self.calibrate_from_folder():
            print("\n❌ 캘리브레이션 실패!")
            return
        
        # 2단계: UDP 수신기 초기화
        self.init_receivers()
        
        # 3단계: 실시간 스티칭
        print("\n" + "=" * 60)
        print("실시간 스티칭 시작 (q 키로 종료)")
        print("=" * 60)
        
        frame_idx = 0
        success_count = 0
        
        # 비디오 저장
        video_writer = None
        
        if self.args.save_images:
            os.makedirs(self.args.save_images, exist_ok=True)
            print(f"이미지 저장 디렉토리: {self.args.save_images}")
        
        try:
            while True:
                # 프레임 수신
                images_8cam = self.get_8cam_images()
                
                if images_8cam is None:
                    continue
                    
                # 프레임 스킵
                self.frame_skip += 1
                if self.frame_skip % self.skip_interval != 0:
                    continue
                    
                # 스티칭
                start_time = time.time()
                pano = self.stitch(images_8cam)
                stitch_time = time.time() - start_time
                
                if pano is None:
                    continue
                    
                success_count += 1
                
                # FPS 계산
                self.stitch_count += 1
                current_time = time.time()
                if current_time - self.stitch_time >= 1.0:
                    self.stitch_fps = self.stitch_count / (current_time - self.stitch_time)
                    self.stitch_count = 0
                    self.stitch_time = current_time
                    
                # 정보 표시
                info_text = f"FPS: {self.stitch_fps:.1f} | Frame: {frame_idx} | Hybrid v6"
                for i, receiver in enumerate(self.receivers):
                    info_text += f" | Cam{i+1}: {receiver.fps:.1f}"
                    
                cv2.putText(pano, info_text, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                           
                # 비디오 저장 초기화
                if video_writer is None and self.args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        self.args.save_video,
                        fourcc,
                        self.args.fps,
                        (pano.shape[1], pano.shape[0])
                    )
                    print(f"\n비디오 저장 시작: {self.args.save_video}")
                
                # 화면 표시
                display_height = 600
                display_width = int(display_height * pano.shape[1] / pano.shape[0])
                display = cv2.resize(pano, (display_width, display_height))
                cv2.imshow('Hybrid Panorama v6', display)
                
                # 비디오 저장
                if video_writer:
                    video_writer.write(pano)
                
                # 이미지 저장
                if self.args.save_images:
                    img_path = os.path.join(self.args.save_images, f'panorama_{frame_idx:04d}.jpg')
                    cv2.imwrite(img_path, pano)
                    
                if (frame_idx + 1) % 10 == 0:
                    print(f"  프레임 {frame_idx+1} | FPS: {self.stitch_fps:.1f}")
                    
                frame_idx += 1
                
                # 종료
                wait_time = max(1, int(1000 / self.args.fps) - int((time.time() - start_time) * 1000))
                key = cv2.waitKey(wait_time)
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n\n중단됨")
            
        finally:
            # 정리
            for receiver in self.receivers:
                receiver.stop()
                
            if video_writer:
                video_writer.release()
                
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print("통계")
            print("=" * 60)
            print(f"성공: {success_count} 프레임")
            print(f"평균 FPS: {self.stitch_fps:.2f}")
            
            if self.args.save_video and video_writer:
                print(f"\n✅ 비디오 저장: {self.args.save_video}")
            
            if self.args.save_images:
                print(f"\n✅ 이미지 저장: {self.args.save_images}")
                print(f"   총 {success_count}장")


def main():
    parser = argparse.ArgumentParser(description="하이브리드 파노라마 스티칭 v6")
    
    # 캘리브레이션 설정 (폴더)
    parser.add_argument("--calibration_dir", type=str, required=True,
                       help="캘리브레이션용 폴더 경로")
    parser.add_argument("--calibration_frames", type=int, default=10)
    parser.add_argument("--reference_frame", type=int, default=7)
    
    # UDP 설정
    parser.add_argument("--ports", type=int, nargs="+", default=[5001, 5002])
    
    # 카메라 설정
    parser.add_argument("--camera_order", type=int, nargs="+", default=None)
    parser.add_argument("--crop_edges", type=int, default=0)
    parser.add_argument("--crop_left", type=int, default=0)
    parser.add_argument("--crop_right", type=int, default=0)
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    
    # 스티칭 설정
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--target_fps", type=int, default=10)
    
    # 저장 설정
    parser.add_argument("--save_images", type=str, default=None)
    parser.add_argument("--save_video", type=str, default=None)
    parser.add_argument("--save_reference", type=str, default="reference_hybrid_v6.jpg")
    parser.add_argument("--fps", type=int, default=10)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("하이브리드 파노라마 스티칭 v6")
    print("=" * 60)
    print(f"캘리브레이션 폴더: {args.calibration_dir}")
    print(f"기준 프레임: {args.reference_frame}")
    print(f"UDP 포트: {args.ports}")
    print(f"카메라 수: {len(args.ports) * 4} (2x2 분할)")
    print(f"스케일: {args.scale * 100:.0f}%")
    print(f"목표 FPS: {args.target_fps}")
    if args.camera_order:
        print(f"카메라 순서: {args.camera_order}")
    print("=" * 60)
    
    stitcher = HybridStitcherV6(args)
    stitcher.run()


if __name__ == "__main__":
    main()
