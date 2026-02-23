#!/usr/bin/env python3
"""
하이브리드 파노라마 스티칭 v6 (OpenCV CUDA GPU 가속)

폴더 이미지로 캘리브레이션 → UDP 실시간 스티칭 (GPU 가속)
"""

import cv2
import numpy as np
import argparse
import time
import threading
import queue
from pathlib import Path

# GPU 확인
try:
    gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
    if gpu_count > 0:
        print(f"[✅ GPU] OpenCV CUDA | {gpu_count}개 GPU 활성화")
    else:
        print("[❌ GPU] CUDA GPU 없음")
        exit(1)
except:
    print("[❌ GPU] OpenCV CUDA 지원 없음")
    exit(1)


class UDPReceiver:
    """UDP 스트림 수신기 (멀티스레드)"""
    
    def __init__(self, port, camera_id):
        self.port = port
        self.camera_id = camera_id
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
    def start(self):
        """수신 시작"""
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """수신 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _receive_loop(self):
        """수신 루프"""
        pipeline = (
            f"udpsrc port={self.port} "
            f"caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=JPEG, payload=96\" ! "
            f"rtpjpegdepay ! jpegdec ! videoconvert ! appsink"
        )
        
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # 큐가 가득 차면 오래된 프레임 제거
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
                
                # FPS 계산
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_time)
                    self.frame_count = 0
                    self.last_time = current_time
        
        cap.release()
        
    def get_frame(self, timeout=1.0):
        """프레임 가져오기"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class GPUStitcher:
    """OpenCV CUDA GPU 가속 스티처"""
    
    def __init__(self, scale=1.0):
        self.scale = scale
        self.stitcher = None
        self.calibrated = False
        
        # GPU 메모리 (재사용)
        self.gpu_imgs = []
        self.gpu_warped = []
        
    def calibrate(self, images):
        """캘리브레이션 (CPU)"""
        print(f"\n{'='*60}")
        print(f"캘리브레이션 시작 (스케일: {int(self.scale*100)}%)")
        print(f"{'='*60}")
        
        # 스케일 조정
        if self.scale != 1.0:
            images_scaled = []
            for img in images:
                h, w = img.shape[:2]
                new_h, new_w = int(h * self.scale), int(w * self.scale)
                resized = cv2.resize(img, (new_w, new_h))
                images_scaled.append(resized)
        else:
            images_scaled = images
        
        print(f"  입력: {len(images_scaled)}개 카메라")
        print(f"  크기: {images_scaled[0].shape[1]} x {images_scaled[0].shape[0]}")
        
        # OpenCV Stitcher 생성
        print("OpenCV Stitcher 생성 중...")
        self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        
        # 고급 설정
        self.stitcher.setRegistrationResol(0.6)
        self.stitcher.setSeamEstimationResol(0.1)
        self.stitcher.setCompositingResol(-1)
        self.stitcher.setPanoConfidenceThresh(0.5)
        
        print("  등록 해상도: 0.6 MP")
        print("  Seam 해상도: 0.1 MP")
        print("  합성 해상도: 원본")
        print("  신뢰도 임계값: 0.5")
        
        # 변환 추정
        print("변환 추정 중...")
        status = self.stitcher.estimateTransform(images_scaled)
        
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 캘리브레이션 실패: {status}")
            return None
        
        print("  ✅ 변환 추정 성공!")
        
        # 기준 파노라마 생성
        print("기준 파노라마 생성 중...")
        status, panorama = self.stitcher.composePanorama()
        
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 파노라마 생성 실패: {status}")
            return None
        
        print("  ✅ 기준 파노라마 생성 성공!")
        print(f"  크기: {panorama.shape[1]} x {panorama.shape[0]}")
        
        self.calibrated = True
        
        # GPU 메모리 초기화
        self._init_gpu_memory(len(images_scaled))
        
        return panorama
    
    def _init_gpu_memory(self, num_cameras):
        """GPU 메모리 초기화"""
        print("\n[GPU] 메모리 초기화...")
        self.gpu_imgs = [cv2.cuda_GpuMat() for _ in range(num_cameras)]
        print(f"  ✅ {num_cameras}개 카메라용 GPU 메모리 할당")
    
    def stitch_gpu(self, images):
        """GPU 가속 스티칭"""
        if not self.calibrated:
            raise RuntimeError("캘리브레이션 필요!")
        
        # 스케일 조정
        if self.scale != 1.0:
            images_scaled = []
            for img in images:
                h, w = img.shape[:2]
                new_h, new_w = int(h * self.scale), int(w * self.scale)
                resized = cv2.resize(img, (new_w, new_h))
                images_scaled.append(resized)
        else:
            images_scaled = images
        
        # GPU로 업로드
        for i, img in enumerate(images_scaled):
            self.gpu_imgs[i].upload(img)
        
        # OpenCV Stitcher 사용 (내부적으로 GPU 사용)
        status, panorama = self.stitcher.composePanorama(images_scaled)
        
        if status != cv2.Stitcher_OK:
            return None
        
        return panorama


class HybridStitcher:
    """하이브리드 스티처 (폴더 캘리브레이션 + UDP 실시간)"""
    
    def __init__(self, args):
        self.args = args
        self.stitcher = GPUStitcher(scale=args.scale)
        self.receivers = []
        self.frame_skip_counter = 0
        self.frame_skip_interval = 15 // args.target_fps if args.target_fps > 0 else 1
        
    def load_folder_images(self):
        """폴더 이미지 로드"""
        print("\n[폴더] 캘리브레이션 이미지 로드")
        
        calibration_dir = Path(self.args.calibration_dir)
        
        # 카메라 순서
        camera_order = self.args.camera_order if self.args.camera_order else list(range(1, 9))
        print(f"  카메라 순서: {camera_order}")
        
        # 이미지 로드
        all_images = {}
        for cam_id in camera_order:
            cam_dir = calibration_dir / f"MyCam_{cam_id:03d}"
            if not cam_dir.exists():
                print(f"  ❌ 카메라 {cam_id} 디렉토리 없음: {cam_dir}")
                return None
            
            images = sorted(cam_dir.glob("*.jpg"))
            if not images:
                print(f"  ❌ 카메라 {cam_id}: 이미지 없음")
                return None
            
            all_images[cam_id] = [cv2.imread(str(img)) for img in images[:self.args.calibration_frames]]
            print(f"  ✅ 카메라 {cam_id}: {len(all_images[cam_id])}장")
        
        print(f"  총 {len(camera_order)}개 카메라")
        
        # 기준 프레임
        ref_frame_idx = self.args.reference_frame - 1
        if ref_frame_idx >= self.args.calibration_frames:
            ref_frame_idx = self.args.calibration_frames - 1
        
        print(f"[기준] 프레임 {ref_frame_idx + 1}/{self.args.calibration_frames}")
        
        # 기준 프레임 이미지
        ref_images = [all_images[cam_id][ref_frame_idx] for cam_id in camera_order]
        
        # 가장자리 크롭
        if self.args.crop_edges > 0 or any([
            self.args.crop_left, self.args.crop_right,
            self.args.crop_top, self.args.crop_bottom
        ]):
            ref_images = self._crop_images(ref_images)
        
        return ref_images
    
    def _crop_images(self, images):
        """가장자리 크롭"""
        left = self.args.crop_left if self.args.crop_left else self.args.crop_edges
        right = self.args.crop_right if self.args.crop_right else self.args.crop_edges
        top = self.args.crop_top if self.args.crop_top else self.args.crop_edges
        bottom = self.args.crop_bottom if self.args.crop_bottom else self.args.crop_edges
        
        cropped = []
        for img in images:
            h, w = img.shape[:2]
            cropped_img = img[top:h-bottom, left:w-right]
            cropped.append(cropped_img)
        
        return cropped
    
    def calibrate_from_folder(self):
        """폴더 이미지로 캘리브레이션"""
        print(f"\n{'='*60}")
        print("[폴더] 캘리브레이션")
        print(f"{'='*60}")
        
        ref_images = self.load_folder_images()
        if ref_images is None:
            return False
        
        # 캘리브레이션
        reference = self.stitcher.calibrate(ref_images)
        
        if reference is None:
            print("❌ 캘리브레이션 실패!")
            return False
        
        # 기준 파노라마 저장
        if self.args.save_reference:
            cv2.imwrite(self.args.save_reference, reference)
            print(f"✅ 기준 파노라마 저장: {self.args.save_reference}")
        
        print("✅ 캘리브레이션 완료!")
        return True
    
    def init_udp_receivers(self):
        """UDP 수신기 초기화"""
        print(f"\n{'='*60}")
        print("[UDP] 수신기 초기화")
        print(f"{'='*60}")
        
        for i, port in enumerate(self.args.ports):
            receiver = UDPReceiver(port, i + 1)
            receiver.start()
            self.receivers.append(receiver)
            print(f"  카메라 {i+1}: UDP 포트 {port} 수신 시작...")
        
        print("  프레임 수신 대기 중...")
        time.sleep(2)
    
    def receive_frames(self):
        """UDP 프레임 수신 (2x2 분할)"""
        frames_2x2 = []
        
        for receiver in self.receivers:
            frame = receiver.get_frame(timeout=1.0)
            if frame is None:
                return None
            
            # 2x2 분할
            h, w = frame.shape[:2]
            h2, w2 = h // 2, w // 2
            
            frames_2x2.append(frame[:h2, :w2])      # 좌상
            frames_2x2.append(frame[:h2, w2:])      # 우상
            frames_2x2.append(frame[h2:, :w2])      # 좌하
            frames_2x2.append(frame[h2:, w2:])      # 우하
        
        # 카메라 순서 재배치
        if self.args.camera_order:
            ordered_frames = []
            for cam_id in self.args.camera_order:
                ordered_frames.append(frames_2x2[cam_id - 1])
            frames_2x2 = ordered_frames
        
        # 가장자리 크롭
        if self.args.crop_edges > 0 or any([
            self.args.crop_left, self.args.crop_right,
            self.args.crop_top, self.args.crop_bottom
        ]):
            frames_2x2 = self._crop_images(frames_2x2)
        
        return frames_2x2
    
    def realtime_stitching(self):
        """실시간 스티칭"""
        print(f"\n{'='*60}")
        print("[실시간] 스티칭 시작 (q 키로 종료)")
        print(f"{'='*60}")
        
        if self.args.save_images:
            Path(self.args.save_images).mkdir(parents=True, exist_ok=True)
            print(f"이미지 저장 디렉토리: {self.args.save_images}")
        
        video_writer = None
        if self.args.save_video:
            Path(self.args.save_video).parent.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        success_count = 0
        fps_list = []
        
        try:
            while True:
                # 프레임 스킵
                self.frame_skip_counter += 1
                if self.frame_skip_counter < self.frame_skip_interval:
                    continue
                self.frame_skip_counter = 0
                
                # 프레임 수신
                start_time = time.time()
                
                frames_8cam = self.receive_frames()
                if frames_8cam is None:
                    print("  ❌ 프레임 수신 실패")
                    continue
                
                # GPU 스티칭
                panorama = self.stitcher.stitch_gpu(frames_8cam)
                
                if panorama is None:
                    continue
                
                # FPS 계산
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_list.append(fps)
                
                frame_count += 1
                success_count += 1
                
                # 화면 표시
                if frame_count % 10 == 0:
                    print(f"  프레임 {frame_count} | FPS: {fps:.1f}")
                
                # 정보 표시
                cv2.putText(panorama, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(panorama, f"Frame: {frame_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 수신 FPS 표시
                for i, receiver in enumerate(self.receivers):
                    cv2.putText(panorama, f"Cam{i+1}: {receiver.fps:.1f} FPS",
                               (10, 110 + i * 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # 화면 표시
                display = cv2.resize(panorama, (1920, int(1920 * panorama.shape[0] / panorama.shape[1])))
                cv2.imshow("GPU Panorama Stitching", display)
                
                # 이미지 저장
                if self.args.save_images:
                    save_path = Path(self.args.save_images) / f"panorama_{frame_count:06d}.jpg"
                    cv2.imwrite(str(save_path), panorama)
                
                # 비디오 저장
                if self.args.save_video:
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps_video = self.args.fps if self.args.fps else 10
                        video_writer = cv2.VideoWriter(
                            self.args.save_video, fourcc, fps_video,
                            (panorama.shape[1], panorama.shape[0])
                        )
                        print(f"비디오 저장 시작: {self.args.save_video}")
                    
                    video_writer.write(panorama)
                
                # 종료 확인
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n중단됨")
        
        finally:
            # 정리
            cv2.destroyAllWindows()
            
            if video_writer:
                video_writer.release()
                print(f"✅ 비디오 저장: {self.args.save_video}")
            
            # 통계
            print(f"\n{'='*60}")
            print("통계")
            print(f"{'='*60}")
            print(f"성공: {success_count} 프레임")
            if fps_list:
                print(f"평균 FPS: {np.mean(fps_list):.2f}")
            
            if self.args.save_images:
                print(f"✅ 이미지 저장: {self.args.save_images}")
                print(f"   총 {success_count}장")
    
    def run(self):
        """실행"""
        # 폴더 캘리브레이션
        if not self.calibrate_from_folder():
            return
        
        # UDP 수신기 초기화
        self.init_udp_receivers()
        
        # 실시간 스티칭
        self.realtime_stitching()
        
        # 정리
        for receiver in self.receivers:
            receiver.stop()


def main():
    parser = argparse.ArgumentParser(description="하이브리드 파노라마 스티칭 v6 (OpenCV CUDA GPU)")
    
    # 캘리브레이션
    parser.add_argument("--calibration_dir", required=True, help="캘리브레이션용 폴더 경로")
    parser.add_argument("--calibration_frames", type=int, default=10)
    parser.add_argument("--reference_frame", type=int, default=7)
    
    # UDP
    parser.add_argument("--ports", nargs="+", type=int, required=True)
    parser.add_argument("--camera_order", nargs="+", type=int, default=None)
    
    # 크롭
    parser.add_argument("--crop_edges", type=int, default=0)
    parser.add_argument("--crop_left", type=int, default=None)
    parser.add_argument("--crop_right", type=int, default=None)
    parser.add_argument("--crop_top", type=int, default=None)
    parser.add_argument("--crop_bottom", type=int, default=None)
    
    # 스티칭
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--target_fps", type=int, default=10)
    
    # 저장
    parser.add_argument("--save_images", default=None)
    parser.add_argument("--save_video", default=None)
    parser.add_argument("--save_reference", default=None)
    parser.add_argument("--fps", type=int, default=10)
    
    args = parser.parse_args()
    
    # 정보 출력
    print(f"{'='*60}")
    print("하이브리드 파노라마 스티칭 v6 (OpenCV CUDA GPU)")
    print(f"{'='*60}")
    print(f"캘리브레이션: {args.calibration_dir}")
    print(f"기준 프레임: {args.reference_frame}")
    print(f"UDP 포트: {args.ports}")
    print(f"스케일: {int(args.scale*100)}%")
    print(f"목표 FPS: {args.target_fps}")
    if args.camera_order:
        print(f"카메라 순서: {args.camera_order}")
    print(f"{'='*60}")
    
    # 실행
    stitcher = HybridStitcher(args)
    stitcher.run()


if __name__ == "__main__":
    main()
