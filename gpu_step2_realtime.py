#!/usr/bin/env python3
"""
GPU Step 2: CUDA GPU remap 기반 실시간 스티칭
- NPZ에서 워핑 맵(xmap, ymap) 로드
- cv2.cuda.remap()으로 GPU 가속 워핑
- 성능 측정 및 CSV 저장
- 목표: 10+ FPS
"""

import os
import argparse
import time
import csv
import numpy as np
import cv2 as cv
import threading
from queue import Queue, Empty
from datetime import datetime


# OpenCV CUDA 확인
try:
    gpu_count = cv.cuda.getCudaEnabledDeviceCount()
    CUDA_AVAILABLE = gpu_count > 0
    if CUDA_AVAILABLE:
        print(f"[✅ OpenCV CUDA] {gpu_count}개 GPU 활성화")
    else:
        print("[⚠️ OpenCV CUDA] GPU 없음 → CPU 모드")
except:
    print("[⚠️ OpenCV CUDA] CUDA 지원 없음 → CPU 모드")
    CUDA_AVAILABLE = False


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
        print(f"  {self.name}: UDP 포트 {self.port}")
        
    def _receive_loop(self):
        pipeline = (
            f"udpsrc port={self.port} "
            f"caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=JPEG, payload=96\" ! "
            f"rtpjpegdepay ! jpegdec ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink sync=false drop=true max-buffers=1"
        )
        
        cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
        
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


class GPURealtimeStitcher:
    """CUDA GPU remap 기반 실시간 스티칭"""
    
    def __init__(self, args):
        self.args = args
        self.calibration = None
        self.receivers = []
        self.use_cuda = CUDA_AVAILABLE
        
        # 워핑 맵
        self.num_cameras = 0
        self.xmaps = []
        self.ymaps = []
        self.masks = []
        self.corners = []
        self.sizes = []
        
        # GPU 메모리 (사전 할당)
        self.gpu_xmaps = []
        self.gpu_ymaps = []
        
        # 파노라마 정보
        self.pano_x = 0
        self.pano_y = 0
        self.pano_w = 0
        self.pano_h = 0
        
        # 성능 측정
        self.performance_data = []
        self.stitch_fps = 0
        self.stitch_count = 0
        self.stitch_time = time.time()
    
    def load_calibration(self):
        """NPZ 파일에서 캘리브레이션 로드"""
        print("=" * 60)
        print("[NPZ] 캘리브레이션 로드")
        print("=" * 60)
        
        if not os.path.exists(self.args.calibration_npz):
            print(f"❌ NPZ 파일 없음: {self.args.calibration_npz}")
            return False
        
        data = np.load(self.args.calibration_npz, allow_pickle=True)
        self.calibration = data
        
        self.num_cameras = int(data['num_cameras'])
        self.pano_x = int(data['pano_x'])
        self.pano_y = int(data['pano_y'])
        self.pano_w = int(data['pano_width'])
        self.pano_h = int(data['pano_height'])
        
        print(f"  ✅ NPZ 로드: {self.args.calibration_npz}")
        print(f"  스케일: {data['scale'] * 100:.0f}%")
        print(f"  카메라: {self.num_cameras}개")
        print(f"  카메라 순서: {data['camera_order'].tolist()}")
        print(f"  이미지 크기: {data['image_width']}x{data['image_height']}")
        print(f"  파노라마 크기: {self.pano_w}x{self.pano_h}")
        
        # 워핑 맵 로드
        for i in range(self.num_cameras):
            self.xmaps.append(data[f'xmap_{i}'].astype(np.float32))
            self.ymaps.append(data[f'ymap_{i}'].astype(np.float32))
            self.masks.append(data[f'mask_{i}'])
            self.corners.append(data[f'corner_{i}'].tolist())
            self.sizes.append(data[f'size_{i}'].tolist())
        
        print("✅ 캘리브레이션 로드 완료!\n")
        return True
    
    def init_gpu(self):
        """GPU 메모리 초기화"""
        if not self.use_cuda:
            print("[⚠️] CUDA 없음 → CPU 모드\n")
            return True
        
        print("=" * 60)
        print("[GPU] 메모리 초기화")
        print("=" * 60)
        
        for i in range(self.num_cameras):
            gpu_xmap = cv.cuda_GpuMat()
            gpu_xmap.upload(self.xmaps[i])
            self.gpu_xmaps.append(gpu_xmap)
            
            gpu_ymap = cv.cuda_GpuMat()
            gpu_ymap.upload(self.ymaps[i])
            self.gpu_ymaps.append(gpu_ymap)
        
        print(f"  ✅ {self.num_cameras}개 워핑 맵 GPU 업로드")
        print("✅ GPU 초기화 완료!\n")
        return True
    
    def init_receivers(self):
        """UDP 수신기 초기화"""
        print("=" * 60)
        print("[UDP] 수신기 초기화")
        print("=" * 60)
        
        for i, port in enumerate(self.args.ports):
            receiver = UDPReceiver(port, f"카메라 {i+1}")
            receiver.start()
            self.receivers.append(receiver)
        
        print("\n  프레임 수신 대기 중...")
        time.sleep(2)
        print("✅ UDP 수신기 초기화 완료!\n")
    
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
        
        # 순서
        camera_order = self.calibration['camera_order'].tolist()
        images_8cam = [images_8cam[i-1] for i in camera_order]
        
        # 리사이즈
        scale = float(self.calibration['scale'])
        if scale != 1.0:
            images_8cam = [
                cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
                for img in images_8cam
            ]
        
        return images_8cam
    
    def stitch_gpu(self, images):
        """CUDA GPU remap으로 워핑 + 블렌딩"""
        # 파노라마 버퍼
        result = np.zeros((self.pano_h, self.pano_w, 3), dtype=np.float32)
        weight_sum = np.zeros((self.pano_h, self.pano_w), dtype=np.float32)
        
        for i in range(self.num_cameras):
            # GPU 업로드
            gpu_img = cv.cuda_GpuMat(images[i])
            
            # GPU remap (핵심 가속!) - interpolation 키워드 인자 사용
            gpu_warped = cv.cuda.remap(gpu_img, self.gpu_xmaps[i], self.gpu_ymaps[i],
                                        interpolation=cv.INTER_LINEAR,
                                        borderMode=cv.BORDER_CONSTANT)
            
            # CPU로 다운로드
            warped = gpu_warped.download()
            
            # 마스크
            mask = self.masks[i].astype(np.float32) / 255.0
            
            # 파노라마 좌표 계산
            cx, cy = self.corners[i]
            ww, wh = self.sizes[i]
            
            # 오프셋 적용
            px = cx - self.pano_x
            py = cy - self.pano_y
            
            # 범위 클리핑
            src_y_start = max(0, -py)
            src_x_start = max(0, -px)
            dst_y_start = max(0, py)
            dst_x_start = max(0, px)
            
            copy_h = min(wh - src_y_start, self.pano_h - dst_y_start)
            copy_w = min(ww - src_x_start, self.pano_w - dst_x_start)
            
            if copy_h <= 0 or copy_w <= 0:
                continue
            
            # 워핑된 이미지 영역
            warped_region = warped[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
            mask_region = mask[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
            
            # 누적
            result[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] += \
                warped_region.astype(np.float32) * mask_region[:, :, np.newaxis]
            weight_sum[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] += mask_region
        
        # 정규화
        valid = weight_sum > 0
        result[valid] /= weight_sum[valid, np.newaxis]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def stitch_cpu(self, images):
        """CPU remap으로 워핑 + 블렌딩 (폴백)"""
        result = np.zeros((self.pano_h, self.pano_w, 3), dtype=np.float32)
        weight_sum = np.zeros((self.pano_h, self.pano_w), dtype=np.float32)
        
        for i in range(self.num_cameras):
            # CPU remap
            warped = cv.remap(images[i], self.xmaps[i], self.ymaps[i],
                             cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
            
            mask = self.masks[i].astype(np.float32) / 255.0
            
            cx, cy = self.corners[i]
            ww, wh = self.sizes[i]
            
            px = cx - self.pano_x
            py = cy - self.pano_y
            
            src_y_start = max(0, -py)
            src_x_start = max(0, -px)
            dst_y_start = max(0, py)
            dst_x_start = max(0, px)
            
            copy_h = min(wh - src_y_start, self.pano_h - dst_y_start)
            copy_w = min(ww - src_x_start, self.pano_w - dst_x_start)
            
            if copy_h <= 0 or copy_w <= 0:
                continue
            
            warped_region = warped[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
            mask_region = mask[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
            
            result[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] += \
                warped_region.astype(np.float32) * mask_region[:, :, np.newaxis]
            weight_sum[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] += mask_region
        
        valid = weight_sum > 0
        result[valid] /= weight_sum[valid, np.newaxis]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def save_performance_csv(self):
        """성능 데이터를 CSV로 저장"""
        if not self.performance_data:
            return
        
        csv_path = self.args.performance_csv
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Frame', 'Timestamp', 'Processing_Time_ms', 'Instantaneous_FPS',
                'UDP1_FPS', 'UDP2_FPS', 'Mode', 'Success'
            ])
            for row in self.performance_data:
                writer.writerow(row)
        
        print(f"\n✅ 성능 데이터 저장: {csv_path}")
        
        # 통계
        processing_times = [row[2] for row in self.performance_data if row[7]]
        if processing_times:
            avg_time = np.mean(processing_times)
            min_time = np.min(processing_times)
            max_time = np.max(processing_times)
            std_time = np.std(processing_times)
            avg_fps = 1000.0 / avg_time if avg_time > 0 else 0
            
            print(f"\n{'='*60}")
            print("성능 통계")
            print(f"{'='*60}")
            print(f"모드: {'CUDA GPU' if self.use_cuda else 'CPU'}")
            print(f"총 프레임: {len(self.performance_data)}")
            print(f"성공: {sum(1 for row in self.performance_data if row[7])}개")
            print(f"평균 처리 시간: {avg_time:.2f} ms (±{std_time:.2f})")
            print(f"최소 처리 시간: {min_time:.2f} ms")
            print(f"최대 처리 시간: {max_time:.2f} ms")
            print(f"평균 FPS: {avg_fps:.2f}")
            
            if avg_fps >= self.args.target_fps:
                print(f"✅ 목표 FPS {self.args.target_fps} 달성!")
            else:
                print(f"⚠️ 목표 FPS {self.args.target_fps} 미달성 (현재: {avg_fps:.2f})")
            
            print(f"{'='*60}")
    
    def run(self):
        """실행"""
        if not self.load_calibration():
            return
        
        if not self.init_gpu():
            return
        
        self.init_receivers()
        
        print("=" * 60)
        print("[실시간] GPU remap 스티칭 시작 (q 키로 종료)")
        print("=" * 60)
        mode = "CUDA GPU" if self.use_cuda else "CPU"
        print(f"모드: {mode}")
        print(f"목표 FPS: {self.args.target_fps}")
        print(f"스케일: {self.calibration['scale'] * 100:.0f}%")
        print(f"파노라마: {self.pano_w}x{self.pano_h}\n")
        
        frame_idx = 0
        success_count = 0
        video_writer = None
        
        if self.args.save_images:
            os.makedirs(self.args.save_images, exist_ok=True)
        
        if self.args.save_video:
            os.makedirs(os.path.dirname(self.args.save_video) if os.path.dirname(self.args.save_video) else '.', exist_ok=True)
        
        try:
            while True:
                images = self.get_8cam_images()
                if images is None:
                    continue
                
                # 스티칭 (시간 측정)
                start_time = time.time()
                try:
                    if self.use_cuda:
                        pano = self.stitch_gpu(images)
                    else:
                        pano = self.stitch_cpu(images)
                    success = True
                except Exception as e:
                    print(f"  ⚠️ 스티칭 오류: {e}")
                    success = False
                    pano = None
                
                processing_time = (time.time() - start_time) * 1000
                
                # 성능 기록
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                instantaneous_fps = 1000.0 / processing_time if processing_time > 0 else 0
                udp1_fps = self.receivers[0].fps if len(self.receivers) > 0 else 0
                udp2_fps = self.receivers[1].fps if len(self.receivers) > 1 else 0
                
                self.performance_data.append([
                    frame_idx, timestamp, round(processing_time, 2),
                    round(instantaneous_fps, 2),
                    round(udp1_fps, 2), round(udp2_fps, 2),
                    mode, success
                ])
                
                if not success or pano is None:
                    frame_idx += 1
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
                info = f"{mode} | FPS: {self.stitch_fps:.1f} | Frame: {frame_idx} | {processing_time:.0f}ms"
                for i, r in enumerate(self.receivers):
                    info += f" | UDP{i+1}: {r.fps:.1f}"
                
                cv.putText(pano, info, (20, 50),
                          cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # 비디오 저장
                if video_writer is None and self.args.save_video:
                    fourcc = cv.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv.VideoWriter(
                        self.args.save_video, fourcc, self.args.fps,
                        (pano.shape[1], pano.shape[0])
                    )
                    print(f"[비디오] 저장 시작: {self.args.save_video}")
                
                # 화면
                display_h = 600
                display_w = int(display_h * pano.shape[1] / pano.shape[0])
                display = cv.resize(pano, (display_w, display_h))
                cv.imshow('Panorama (GPU remap)', display)
                
                if video_writer:
                    video_writer.write(pano)
                
                if self.args.save_images:
                    cv.imwrite(os.path.join(self.args.save_images, f'pano_{frame_idx:04d}.jpg'), pano)
                
                if (frame_idx + 1) % 10 == 0:
                    print(f"  프레임 {frame_idx+1} | {self.stitch_fps:.1f} FPS | {processing_time:.0f} ms")
                
                frame_idx += 1
                
                if cv.waitKey(1) == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n\n중단됨")
        
        finally:
            for r in self.receivers:
                r.stop()
            
            if video_writer:
                video_writer.release()
            
            cv.destroyAllWindows()
            self.save_performance_csv()
            
            if self.args.save_video:
                print(f"✅ 비디오: {self.args.save_video}")
            if self.args.save_images:
                print(f"✅ 이미지: {self.args.save_images} ({success_count}장)")


def main():
    parser = argparse.ArgumentParser(description="GPU Step 2: CUDA GPU remap 실시간 스티칭")
    
    parser.add_argument("--calibration_npz", type=str, required=True, help="캘리브레이션 NPZ 파일")
    parser.add_argument("--ports", type=int, nargs="+", default=[5001, 5002], help="UDP 포트")
    parser.add_argument("--target_fps", type=int, default=10, help="목표 FPS")
    parser.add_argument("--save_images", type=str, default=None, help="이미지 저장 디렉토리")
    parser.add_argument("--save_video", type=str, default=None, help="비디오 저장 경로")
    parser.add_argument("--performance_csv", type=str, default="performance_gpu.csv", help="성능 CSV 파일")
    parser.add_argument("--fps", type=int, default=10, help="비디오 FPS")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("파노라마 스티칭 - GPU Step 2: CUDA remap 실시간 스티칭")
    print("=" * 60)
    print(f"캘리브레이션 NPZ: {args.calibration_npz}")
    print(f"UDP 포트: {args.ports}")
    print(f"목표 FPS: {args.target_fps}")
    print(f"성능 CSV: {args.performance_csv}")
    print("=" * 60)
    
    stitcher = GPURealtimeStitcher(args)
    stitcher.run()


if __name__ == "__main__":
    main()
