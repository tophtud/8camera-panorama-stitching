#!/usr/bin/env python3
"""
영상 수집율 시험 평가 (hybrid_stitcher_v6 기반)

핵심:
  - OpenCV Stitcher의 estimateTransform() + composePanorama() 방식 그대로 사용
  - 화면 표시 없음 (성능 최적화)
  - 2분간 1초당 1장 생성 × 5회 반복
  - CSV에 생성 시간 기록
  - 목표 수집율: 90%

사용법:
  python3 collection_rate_test_v6.py \
      --calibration_dir ./calibration_images \
      --calibration_frames 10 \
      --reference_frame 7 \
      --ports 5001 5002 \
      --camera_order 5 4 3 2 1 8 7 6 \
      --scale 1.0 \
      --num_trials 5 \
      --output_dir ./output/collection_test
"""

import cv2
import numpy as np
import argparse
import time
import threading
import queue
import csv
import os
from pathlib import Path
from datetime import datetime

# GPU 확인
try:
    gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
    if gpu_count > 0:
        print(f"[✅ GPU] OpenCV CUDA | {gpu_count}개 GPU 활성화")
    else:
        print("[⚠️ GPU] CUDA GPU 없음 → CPU 모드")
except:
    print("[⚠️ GPU] OpenCV CUDA 지원 없음 → CPU 모드")


# ============================================================
# UDP 수신기
# ============================================================
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
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _receive_loop(self):
        pipeline = (
            f"udpsrc port={self.port} "
            f"caps=\"application/x-rtp, media=video, clock-rate=90000, "
            f"encoding-name=JPEG, payload=96\" ! "
            f"rtpjpegdepay ! jpegdec ! videoconvert ! appsink"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)

                self.frame_count += 1
                now = time.time()
                if now - self.last_time >= 1.0:
                    self.fps = self.frame_count / (now - self.last_time)
                    self.frame_count = 0
                    self.last_time = now

        cap.release()

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ============================================================
# 수집율 시험
# ============================================================
class CollectionRateTest:
    """영상 수집율 시험 평가 (v6 방식)"""

    def __init__(self, args):
        self.args = args
        self.stitcher = None       # cv2.Stitcher 객체
        self.calibrated = False
        self.receivers = []
        self.scale = args.scale

    # ----------------------------------------------------------
    # 캘리브레이션
    # ----------------------------------------------------------
    def load_folder_images(self):
        """폴더 이미지 로드"""
        print("\n[폴더] 캘리브레이션 이미지 로드")

        calibration_dir = Path(self.args.calibration_dir)
        camera_order = self.args.camera_order
        print(f"  카메라 순서: {camera_order}")

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

            all_images[cam_id] = [cv2.imread(str(img))
                                  for img in images[:self.args.calibration_frames]]
            print(f"  ✅ 카메라 {cam_id}: {len(all_images[cam_id])}장")

        print(f"  총 {len(camera_order)}개 카메라")

        ref_idx = min(self.args.reference_frame - 1,
                      self.args.calibration_frames - 1)
        print(f"[기준] 프레임 {ref_idx + 1}/{self.args.calibration_frames}")

        ref_images = [all_images[cam_id][ref_idx] for cam_id in camera_order]
        return ref_images

    def calibrate(self):
        """OpenCV Stitcher 캘리브레이션 (v6 방식 그대로)"""
        print(f"\n{'=' * 60}")
        print("[캘리브레이션] OpenCV Stitcher")
        print(f"{'=' * 60}")

        ref_images = self.load_folder_images()
        if ref_images is None:
            return False

        # 스케일 적용
        if self.scale != 1.0:
            images_scaled = [
                cv2.resize(img, (int(img.shape[1] * self.scale),
                                 int(img.shape[0] * self.scale)))
                for img in ref_images
            ]
        else:
            images_scaled = ref_images

        print(f"  입력: {len(images_scaled)}개 카메라")
        print(f"  크기: {images_scaled[0].shape[1]} x {images_scaled[0].shape[0]}")

        # OpenCV Stitcher 생성 (v6과 동일)
        print("OpenCV Stitcher 생성 중...")
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
        print("변환 추정 중...")
        status = self.stitcher.estimateTransform(images_scaled)
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 변환 추정 실패: {status}")
            return False
        print("  ✅ 변환 추정 성공!")

        # 기준 파노라마 생성
        print("기준 파노라마 생성 중...")
        status, panorama = self.stitcher.composePanorama()
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 파노라마 생성 실패: {status}")
            return False

        print("  ✅ 기준 파노라마 생성 성공!")
        print(f"  크기: {panorama.shape[1]} x {panorama.shape[0]}")

        # 기준 파노라마 저장
        ref_path = os.path.join(self.args.output_dir, "reference.jpg")
        os.makedirs(self.args.output_dir, exist_ok=True)
        cv2.imwrite(ref_path, panorama)
        print(f"  ✅ 기준 파노라마 저장: {ref_path}")

        self.calibrated = True
        print("✅ 캘리브레이션 완료!")
        return True

    # ----------------------------------------------------------
    # UDP
    # ----------------------------------------------------------
    def init_receivers(self):
        """UDP 수신기 초기화"""
        print(f"\n{'=' * 60}")
        print("[UDP] 수신기 초기화")
        print(f"{'=' * 60}")

        for i, port in enumerate(self.args.ports):
            receiver = UDPReceiver(port, i + 1)
            receiver.start()
            self.receivers.append(receiver)
            print(f"  카메라 {i + 1}: UDP 포트 {port} 수신 시작...")

        print("  프레임 수신 대기 중...")
        time.sleep(2)
        print("  ✅ UDP 수신기 준비 완료")

    def stop_receivers(self):
        for r in self.receivers:
            r.stop()

    def get_8cam_images(self):
        """UDP에서 8개 카메라 이미지 (v6 방식 그대로)"""
        frames_2x2 = []

        for receiver in self.receivers:
            frame = receiver.get_frame(timeout=1.0)
            if frame is None:
                return None

            h, w = frame.shape[:2]
            h2, w2 = h // 2, w // 2

            frames_2x2.append(frame[:h2, :w2])      # 좌상
            frames_2x2.append(frame[:h2, w2:])       # 우상
            frames_2x2.append(frame[h2:, :w2])       # 좌하
            frames_2x2.append(frame[h2:, w2:])       # 우하

        # 카메라 순서 재배치
        ordered = [frames_2x2[cam_id - 1] for cam_id in self.args.camera_order]

        # 스케일 적용
        if self.scale != 1.0:
            ordered = [
                cv2.resize(img, (int(img.shape[1] * self.scale),
                                 int(img.shape[0] * self.scale)))
                for img in ordered
            ]

        return ordered

    # ----------------------------------------------------------
    # 스티칭 (v6 방식 그대로)
    # ----------------------------------------------------------
    def stitch(self, images):
        """composePanorama(images) 호출 (v6 방식)"""
        status, panorama = self.stitcher.composePanorama(images)
        if status != cv2.Stitcher_OK:
            return None
        return panorama

    # ----------------------------------------------------------
    # 시험
    # ----------------------------------------------------------
    def run_single_trial(self, trial_num):
        """단일 시험 실행 (2분)"""
        DURATION = 120   # 2분
        INTERVAL = 1.0   # 1초당 1장
        TARGET = 120     # 목표 120장

        trial_dir = os.path.join(self.args.output_dir, f"trial_{trial_num}")
        os.makedirs(trial_dir, exist_ok=True)
        csv_path = os.path.join(trial_dir, f"trial_{trial_num}_result.csv")

        print(f"\n{'=' * 60}")
        print(f"[시험 {trial_num}/{self.args.num_trials}] 시작")
        print(f"{'=' * 60}")
        print(f"  목표: {DURATION}초 동안 {TARGET}장 생성")
        print(f"  간격: {INTERVAL}초")
        print(f"  저장: {trial_dir}")
        print()

        # CSV
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(csv_file)
        writer.writerow([
            'Frame_Number', 'Target_Time_s', 'Actual_Time_s',
            'Generation_Time', 'Stitch_Time_ms', 'Total_Time_ms',
            'Success', 'Image_File'
        ])

        success_count = 0
        fail_count = 0
        total_stitch_time = 0.0

        test_start = time.time()
        next_capture = test_start
        frame_num = 0

        while True:
            now = time.time()
            elapsed = now - test_start

            # 2분 경과 → 종료
            if elapsed >= DURATION:
                print(f"\n  ⏱️ {DURATION}초 경과 → 시험 종료")
                break

            # 1초 간격 타이머
            if now < next_capture:
                time.sleep(0.001)
                continue

            frame_num += 1
            target_time = (frame_num - 1) * INTERVAL
            actual_time = elapsed
            frame_start = time.time()

            # 1. 프레임 수신
            images = self.get_8cam_images()
            if images is None:
                fail_count += 1
                gen_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                writer.writerow([
                    frame_num, f"{target_time:.1f}", f"{actual_time:.3f}",
                    gen_time, 0, 0, False, "NO_FRAME"
                ])
                next_capture += INTERVAL
                continue

            # 2. 스티칭 (composePanorama - v6 방식)
            stitch_start = time.time()
            try:
                pano = self.stitch(images)
            except Exception as e:
                pano = None
                print(f"  ⚠️ 스티칭 오류: {e}")

            stitch_ms = (time.time() - stitch_start) * 1000
            gen_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            if pano is not None:
                # 3. 이미지 저장
                success_count += 1
                total_stitch_time += stitch_ms
                img_filename = f"pano_{frame_num:04d}.jpg"
                img_path = os.path.join(trial_dir, img_filename)
                cv2.imwrite(img_path, pano, [cv2.IMWRITE_JPEG_QUALITY, 85])

                total_ms = (time.time() - frame_start) * 1000

                writer.writerow([
                    frame_num, f"{target_time:.1f}", f"{actual_time:.3f}",
                    gen_time, f"{stitch_ms:.1f}", f"{total_ms:.1f}",
                    True, img_filename
                ])
            else:
                fail_count += 1
                total_ms = (time.time() - frame_start) * 1000
                writer.writerow([
                    frame_num, f"{target_time:.1f}", f"{actual_time:.3f}",
                    gen_time, f"{stitch_ms:.1f}", f"{total_ms:.1f}",
                    False, "STITCH_FAIL"
                ])

            # 진행 상황 (10프레임마다)
            if frame_num % 10 == 0:
                rate = (success_count / frame_num) * 100
                avg_s = total_stitch_time / success_count if success_count > 0 else 0
                udp_fps = [f"UDP{i+1}:{r.fps:.1f}" for i, r in enumerate(self.receivers)]
                print(f"  [{elapsed:.0f}s] 프레임 {frame_num} | "
                      f"성공: {success_count} | 실패: {fail_count} | "
                      f"수집율: {rate:.1f}% | 스티칭: {avg_s:.0f}ms | "
                      f"{' | '.join(udp_fps)}")

            # 다음 캡처 시간
            next_capture += INTERVAL

            # 밀린 프레임 처리
            now2 = time.time()
            while next_capture + INTERVAL < now2 and (now2 - test_start) < DURATION:
                frame_num += 1
                fail_count += 1
                skip_target = (frame_num - 1) * INTERVAL
                skip_gen = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                writer.writerow([
                    frame_num, f"{skip_target:.1f}",
                    f"{now2 - test_start:.3f}",
                    skip_gen, 0, 0, False, "SKIPPED"
                ])
                next_capture += INTERVAL

        csv_file.close()

        # 시험 결과
        total_frames = success_count + fail_count
        collection_rate = (success_count / TARGET) * 100
        avg_stitch = total_stitch_time / success_count if success_count > 0 else 0

        print(f"\n  {'=' * 40}")
        print(f"  [시험 {trial_num}] 결과")
        print(f"  {'=' * 40}")
        print(f"  목표 프레임: {TARGET}장")
        print(f"  시도 프레임: {total_frames}장")
        print(f"  성공 프레임: {success_count}장")
        print(f"  실패 프레임: {fail_count}장")
        print(f"  수집율: {collection_rate:.1f}%")
        print(f"  평균 스티칭: {avg_stitch:.1f} ms")
        if collection_rate >= 90:
            print(f"  ✅ 목표 달성! (90% 이상)")
        else:
            print(f"  ❌ 목표 미달성 (90% 미만)")
        print(f"  CSV: {csv_path}")

        return {
            'trial': trial_num,
            'target': TARGET,
            'total': total_frames,
            'success': success_count,
            'fail': fail_count,
            'rate': collection_rate,
            'avg_stitch_ms': avg_stitch,
        }

    # ----------------------------------------------------------
    # 전체 실행
    # ----------------------------------------------------------
    def run(self):
        """전체 시험 실행"""
        os.makedirs(self.args.output_dir, exist_ok=True)

        # 1. 캘리브레이션
        if not self.calibrate():
            print("\n❌ 캘리브레이션 실패!")
            return

        # 2. UDP 수신기
        self.init_receivers()

        # 3. 시험 실행
        print(f"\n{'=' * 60}")
        print("영상 수집율 시험 시작")
        print(f"{'=' * 60}")
        print(f"  시험 횟수: {self.args.num_trials}회")
        print(f"  시험 시간: 120초 (2분)")
        print(f"  목표: 1초당 1장 (120장/2분)")
        print(f"  목표 수집율: 90%")
        print(f"  스케일: {int(self.scale * 100)}%")
        print(f"  출력: {self.args.output_dir}")

        all_results = []

        for trial in range(1, self.args.num_trials + 1):
            if trial > 1:
                wait = self.args.trial_interval
                print(f"\n⏳ 다음 시험까지 {wait}초 대기...")
                time.sleep(wait)

            result = self.run_single_trial(trial)
            all_results.append(result)

        # 수신기 정지
        self.stop_receivers()

        # 전체 요약 CSV
        summary_csv = os.path.join(self.args.output_dir, "summary.csv")
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'Trial', 'Target', 'Total', 'Success',
                'Fail', 'Collection_Rate_%', 'Avg_Stitch_ms'
            ])
            for r in all_results:
                w.writerow([
                    r['trial'], r['target'], r['total'],
                    r['success'], r['fail'],
                    f"{r['rate']:.1f}",
                    f"{r['avg_stitch_ms']:.1f}"
                ])
            avg_rate = np.mean([r['rate'] for r in all_results])
            avg_stitch = np.mean([r['avg_stitch_ms'] for r in all_results])
            w.writerow([])
            w.writerow(['Average', '', '', '', '',
                        f"{avg_rate:.1f}", f"{avg_stitch:.1f}"])

        # 최종 결과
        print(f"\n{'=' * 60}")
        print("전체 시험 결과 요약")
        print(f"{'=' * 60}")
        print(f"{'시험':>6} | {'목표':>6} | {'성공':>6} | {'실패':>6} | "
              f"{'수집율':>8} | {'스티칭':>10}")
        print("-" * 60)
        for r in all_results:
            s = "✅" if r['rate'] >= 90 else "❌"
            print(f"  {r['trial']:>4} | {r['target']:>5}장 | "
                  f"{r['success']:>5}장 | {r['fail']:>5}장 | "
                  f"{r['rate']:>6.1f}% {s} | {r['avg_stitch_ms']:>8.1f}ms")
        print("-" * 60)
        avg_rate = np.mean([r['rate'] for r in all_results])
        avg_stitch = np.mean([r['avg_stitch_ms'] for r in all_results])
        overall = "✅" if avg_rate >= 90 else "❌"
        print(f"  평균 |       |       |       | "
              f"{avg_rate:>6.1f}% {overall} | {avg_stitch:>8.1f}ms")
        print(f"{'=' * 60}")
        print(f"\n✅ 요약 CSV: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="영상 수집율 시험 평가 (v6 방식)")

    # 캘리브레이션
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--calibration_frames", type=int, default=10)
    parser.add_argument("--reference_frame", type=int, default=7)

    # UDP
    parser.add_argument("--ports", nargs="+", type=int, required=True)
    parser.add_argument("--camera_order", nargs="+", type=int,
                        default=[5, 4, 3, 2, 1, 8, 7, 6])

    # 스티칭
    parser.add_argument("--scale", type=float, default=1.0)

    # 시험
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--trial_interval", type=int, default=10)
    parser.add_argument("--output_dir", type=str,
                        default="./output/collection_test")

    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("영상 수집율 시험 평가 (OpenCV Stitcher v6 방식)")
    print(f"{'=' * 60}")
    print(f"캘리브레이션: {args.calibration_dir}")
    print(f"기준 프레임: {args.reference_frame}")
    print(f"UDP 포트: {args.ports}")
    print(f"카메라 순서: {args.camera_order}")
    print(f"스케일: {int(args.scale * 100)}%")
    print(f"시험 횟수: {args.num_trials}회")
    print(f"시험 시간: 120초 (2분)")
    print(f"목표: 1초당 1장 (120장/2분)")
    print(f"목표 수집율: 90%")
    print(f"출력: {args.output_dir}")
    print(f"{'=' * 60}")

    tester = CollectionRateTest(args)
    tester.run()


if __name__ == "__main__":
    main()
