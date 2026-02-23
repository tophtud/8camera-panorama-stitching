#!/usr/bin/env python3
"""
영상 수집율 시험 평가
- 2분간 1초당 1장 파노라마 생성 (목표: 120장)
- 5회 반복 실시
- CSV에 생성 시간 기록
- 화면 표시 없음 (성능 최적화)
- 목표 수집율: 90% (108장/120장)
"""

import os
import glob
import argparse
import time
import csv
import numpy as np
import cv2
import threading
from queue import Queue, Empty
from datetime import datetime


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


class CollectionRateTest:
    """영상 수집율 시험 평가"""
    
    def __init__(self, args):
        self.args = args
        self.stitcher = None
        self.is_calibrated = False
        self.receivers = []
        
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
        
        all_images = load_camera_images(
            os.path.expanduser(self.args.calibration_dir),
            num_frames=self.args.calibration_frames,
            crop_config=crop_config,
            camera_order=self.args.camera_order
        )
        
        if len(all_images) < 2:
            print("\n❌ 카메라가 부족합니다")
            return False
        
        ref_idx = min(self.args.reference_frame, len(all_images[0]) - 1)
        ref_images = [cam_images[ref_idx] for cam_images in all_images]
        print(f"\n기준 프레임: {ref_idx+1}/{len(all_images[0])}")
        
        images_scaled = [
            cv2.resize(img, None, fx=self.args.scale, fy=self.args.scale, interpolation=cv2.INTER_AREA)
            for img in ref_images
        ]
        
        h, w = images_scaled[0].shape[:2]
        print(f"입력: {len(images_scaled)}개 카메라")
        print(f"크기: {w} x {h}")
        
        print("\nOpenCV Stitcher 생성 중...")
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        
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
        
        print("\n변환 추정 중...")
        status = self.stitcher.estimateTransform(images_scaled)
        
        if status != cv2.Stitcher_OK:
            print(f"❌ 캘리브레이션 실패: {status}")
            return False
        
        print("✅ 변환 추정 성공!")
        
        print("\n기준 파노라마 생성 중...")
        status2, reference = self.stitcher.composePanorama()
        
        if status2 != cv2.Stitcher_OK:
            print(f"❌ 파노라마 생성 실패: {status2}")
            return False
        
        print(f"✅ 기준 파노라마 생성 성공!")
        print(f"크기: {reference.shape[1]} x {reference.shape[0]}")
        
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
    
    def stop_receivers(self):
        """UDP 수신기 정지"""
        for receiver in self.receivers:
            receiver.stop()
        self.receivers = []
    
    def get_8cam_images(self):
        """UDP에서 8개 카메라 이미지 추출"""
        frames_2cam = []
        for receiver in self.receivers:
            frame = receiver.get_frame(timeout=1.0)
            if frame is None:
                return None
            frames_2cam.append(frame)
        
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
        """이미지 스티칭"""
        if not self.is_calibrated:
            return None
        
        images_scaled = [
            cv2.resize(img, None, fx=self.args.scale, fy=self.args.scale, interpolation=cv2.INTER_AREA)
            for img in images
        ]
        
        status, panorama = self.stitcher.composePanorama(images_scaled)
        
        if status != cv2.Stitcher_OK:
            return None
        
        return panorama
    
    def run_single_test(self, trial_num, output_dir):
        """단일 시험 실행 (2분)"""
        test_duration = 120  # 2분 = 120초
        target_interval = 1.0  # 1초당 1장
        target_total = test_duration  # 120장 목표
        
        trial_dir = os.path.join(output_dir, f"trial_{trial_num}")
        os.makedirs(trial_dir, exist_ok=True)
        
        csv_path = os.path.join(trial_dir, f"trial_{trial_num}_result.csv")
        
        print(f"\n{'='*60}")
        print(f"[시험 {trial_num}/5] 시작")
        print(f"{'='*60}")
        print(f"  목표: {test_duration}초 동안 {target_total}장 생성")
        print(f"  간격: {target_interval}초")
        print(f"  저장: {trial_dir}")
        print(f"  CSV: {csv_path}")
        print()
        
        # CSV 파일 생성
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'Frame_Number',       # 프레임 번호
            'Target_Time_s',      # 목표 시간 (초)
            'Actual_Time_s',      # 실제 시간 (초)
            'Generation_Time',    # 생성 시각 (HH:MM:SS.mmm)
            'Processing_ms',      # 처리 시간 (ms)
            'Success',            # 성공 여부
            'Delay_ms',           # 지연 시간 (ms)
            'Image_File'          # 이미지 파일명
        ])
        
        success_count = 0
        fail_count = 0
        total_processing_time = 0
        
        test_start_time = time.time()
        next_capture_time = test_start_time  # 첫 번째 캡처는 즉시
        
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
                # 대기 (CPU 부하 줄이기)
                sleep_time = min(0.01, next_capture_time - current_time)
                time.sleep(sleep_time)
                continue
            
            frame_num += 1
            target_time_s = (frame_num - 1) * target_interval
            actual_time_s = elapsed
            
            # 프레임 수신
            images = self.get_8cam_images()
            
            if images is None:
                # 수신 실패
                fail_count += 1
                gen_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                delay_ms = (actual_time_s - target_time_s) * 1000
                csv_writer.writerow([
                    frame_num, f"{target_time_s:.1f}", f"{actual_time_s:.3f}",
                    gen_time, 0, False, f"{delay_ms:.1f}", ""
                ])
                next_capture_time += target_interval
                continue
            
            # 스티칭
            stitch_start = time.time()
            pano = self.stitch(images)
            processing_ms = (time.time() - stitch_start) * 1000
            total_processing_time += processing_ms
            
            gen_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            delay_ms = (actual_time_s - target_time_s) * 1000
            
            if pano is not None:
                # 성공
                success_count += 1
                img_filename = f"pano_{frame_num:04d}.jpg"
                img_path = os.path.join(trial_dir, img_filename)
                cv2.imwrite(img_path, pano)
                
                csv_writer.writerow([
                    frame_num, f"{target_time_s:.1f}", f"{actual_time_s:.3f}",
                    gen_time, f"{processing_ms:.1f}", True, f"{delay_ms:.1f}", img_filename
                ])
            else:
                # 스티칭 실패
                fail_count += 1
                csv_writer.writerow([
                    frame_num, f"{target_time_s:.1f}", f"{actual_time_s:.3f}",
                    gen_time, f"{processing_ms:.1f}", False, f"{delay_ms:.1f}", ""
                ])
            
            # 진행 상황 (10초마다)
            if frame_num % 10 == 0:
                rate = (success_count / frame_num) * 100 if frame_num > 0 else 0
                print(f"  [{elapsed:.0f}s] 프레임 {frame_num} | 성공: {success_count} | 실패: {fail_count} | 수집율: {rate:.1f}%")
            
            # 다음 캡처 시간 설정
            next_capture_time += target_interval
            
            # 만약 처리가 너무 오래 걸려서 다음 캡처 시간을 이미 지났으면
            # 현재 시간 기준으로 재설정 (밀린 프레임은 건너뜀)
            if time.time() > next_capture_time + target_interval:
                missed = int((time.time() - next_capture_time) / target_interval)
                for m in range(missed):
                    frame_num += 1
                    fail_count += 1
                    skip_time_s = (frame_num - 1) * target_interval
                    skip_gen_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    csv_writer.writerow([
                        frame_num, f"{skip_time_s:.1f}", f"{time.time() - test_start_time:.3f}",
                        skip_gen_time, 0, False, f"{(time.time() - test_start_time - skip_time_s)*1000:.1f}",
                        "SKIPPED"
                    ])
                next_capture_time = time.time() + target_interval
        
        csv_file.close()
        
        # 시험 결과 요약
        total_frames = success_count + fail_count
        collection_rate = (success_count / target_total) * 100 if target_total > 0 else 0
        avg_processing = total_processing_time / success_count if success_count > 0 else 0
        
        print(f"\n  {'='*40}")
        print(f"  [시험 {trial_num}] 결과")
        print(f"  {'='*40}")
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
        """5회 시험 실행"""
        # 1단계: 캘리브레이션
        if not self.calibrate_from_folder():
            print("\n❌ 캘리브레이션 실패!")
            return
        
        # 2단계: UDP 수신기 초기화
        self.init_receivers()
        
        # 출력 디렉토리
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 3단계: 5회 시험 실행
        print("\n" + "=" * 60)
        print("영상 수집율 시험 시작")
        print("=" * 60)
        print(f"  시험 횟수: {self.args.num_trials}회")
        print(f"  시험 시간: 120초 (2분)")
        print(f"  목표 수집율: 90%")
        print(f"  목표 프레임: 1초당 1장 (120장/2분)")
        print(f"  출력 디렉토리: {output_dir}")
        
        all_results = []
        
        for trial in range(1, self.args.num_trials + 1):
            if trial > 1:
                # 시험 간 대기
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
            
            # 평균
            avg_rate = np.mean([r['collection_rate'] for r in all_results])
            avg_proc = np.mean([r['avg_processing_ms'] for r in all_results])
            writer.writerow([])
            writer.writerow(['Average', '', '', '', '', f"{avg_rate:.1f}", f"{avg_proc:.1f}"])
        
        # 최종 결과 출력
        print("\n" + "=" * 60)
        print("전체 시험 결과 요약")
        print("=" * 60)
        print(f"{'시험':>6} | {'목표':>6} | {'성공':>6} | {'실패':>6} | {'수집율':>8} | {'처리시간':>10}")
        print("-" * 60)
        for r in all_results:
            status = "✅" if r['collection_rate'] >= 90 else "❌"
            print(f"  {r['trial']:>4} | {r['target_total']:>5}장 | {r['success']:>5}장 | {r['fail']:>5}장 | {r['collection_rate']:>6.1f}% {status} | {r['avg_processing_ms']:>8.1f}ms")
        print("-" * 60)
        avg_rate = np.mean([r['collection_rate'] for r in all_results])
        avg_proc = np.mean([r['avg_processing_ms'] for r in all_results])
        overall_status = "✅" if avg_rate >= 90 else "❌"
        print(f"  평균 |       |       |       | {avg_rate:>6.1f}% {overall_status} | {avg_proc:>8.1f}ms")
        print("=" * 60)
        print(f"\n✅ 요약 CSV: {summary_csv}")
        print(f"✅ 각 시험 결과: {output_dir}/trial_N/trial_N_result.csv")


def main():
    parser = argparse.ArgumentParser(description="영상 수집율 시험 평가")
    
    # 캘리브레이션 설정
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
    
    # 시험 설정
    parser.add_argument("--num_trials", type=int, default=5, help="시험 횟수 (기본: 5)")
    parser.add_argument("--trial_interval", type=int, default=10, help="시험 간 대기 시간 (초, 기본: 10)")
    
    # 저장 설정
    parser.add_argument("--output_dir", type=str, default="./output/collection_test",
                       help="결과 저장 디렉토리")
    parser.add_argument("--save_reference", type=str, default="reference.jpg")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("영상 수집율 시험 평가")
    print("=" * 60)
    print(f"캘리브레이션 폴더: {args.calibration_dir}")
    print(f"기준 프레임: {args.reference_frame}")
    print(f"UDP 포트: {args.ports}")
    print(f"스케일: {args.scale * 100:.0f}%")
    if args.camera_order:
        print(f"카메라 순서: {args.camera_order}")
    print(f"시험 횟수: {args.num_trials}회")
    print(f"시험 시간: 120초 (2분)")
    print(f"목표: 1초당 1장 (120장/2분)")
    print(f"목표 수집율: 90%")
    print(f"출력: {args.output_dir}")
    print("=" * 60)
    
    tester = CollectionRateTest(args)
    tester.run()


if __name__ == "__main__":
    main()
