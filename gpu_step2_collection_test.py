#!/usr/bin/env python3
"""
GPU Step 2: CuPy GPU 스티칭 + 영상 수집율 시험

전제조건:
  - gpu_step1_calibrate.py로 생성한 NPZ 파일 필요
  - CuPy + CUDA 설치 필요

기능:
  - NPZ에서 워핑 맵(xmap, ymap, mask) 로드
  - CuPy GPU로 워핑 + 가중 블렌딩
  - UDP 스트림 수신 (2포트 × 2x2 분할 = 8카메라)
  - 2분간 1초당 1장 생성 × 5회 반복
  - CSV에 생성 시간 기록
  - 화면 표시 없음 (성능 최적화)
  - 목표 수집율: 90%

사용법:
  python3 gpu_step2_collection_test.py \
      --calibration_npz ./calibration_gpu.npz \
      --ports 5001 5002 \
      --camera_order 5 4 3 2 1 8 7 6 \
      --num_trials 5 \
      --output_dir ./output/collection_test
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

# ============================================================
# CuPy GPU 확인
# ============================================================
try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    if CUPY_AVAILABLE:
        print(f"[✅ CuPy GPU] {cp.cuda.runtime.getDeviceCount()}개 GPU, 버전: {cp.__version__}")
        # GPU 워밍업
        _ = cp.array([1])
        cp.cuda.Stream.null.synchronize()
except ImportError:
    CUPY_AVAILABLE = False
    print("[❌ CuPy 없음] CPU 모드로 실행")


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
        cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
        if not cap.isOpened():
            print(f"  ❌ {self.name}: 스트림 열기 실패!")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_time)
                self.frame_count = 0
                self.last_time = now

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
# CuPy GPU 스티칭 엔진
# ============================================================
class CuPyStitcher:
    """NPZ 기반 CuPy GPU 워핑 + 블렌딩"""

    def __init__(self):
        self.num_cameras = 0
        self.scale = 1.0
        self.img_w = 0
        self.img_h = 0
        self.pano_x = 0
        self.pano_y = 0
        self.pano_w = 0
        self.pano_h = 0
        self.camera_order = []

        # CPU 데이터
        self.xmaps = []
        self.ymaps = []
        self.masks = []
        self.corners = []
        self.sizes = []

        # GPU 데이터
        self.gpu_xmaps = []
        self.gpu_ymaps = []
        self.gpu_masks = []

    def load_npz(self, npz_path):
        """NPZ 파일에서 캘리브레이션 로드"""
        print(f"\n[NPZ] 로드: {npz_path}")

        if not os.path.exists(npz_path):
            print(f"  ❌ 파일 없음: {npz_path}")
            return False

        data = np.load(npz_path, allow_pickle=True)

        self.num_cameras = int(data['num_cameras'])
        self.scale = float(data['scale'])
        self.img_w = int(data['image_width'])
        self.img_h = int(data['image_height'])
        self.pano_x = int(data['pano_x'])
        self.pano_y = int(data['pano_y'])
        self.pano_w = int(data['pano_width'])
        self.pano_h = int(data['pano_height'])
        self.camera_order = data['camera_order'].tolist()

        print(f"  카메라: {self.num_cameras}개")
        print(f"  카메라 순서: {self.camera_order}")
        print(f"  스케일: {self.scale * 100:.0f}%")
        print(f"  입력 크기: {self.img_w}x{self.img_h}")
        print(f"  파노라마: {self.pano_w}x{self.pano_h}")

        # 워핑 맵 로드
        self.xmaps, self.ymaps, self.masks = [], [], []
        self.corners, self.sizes = [], []

        for i in range(self.num_cameras):
            self.xmaps.append(data[f'xmap_{i}'].astype(np.float32))
            self.ymaps.append(data[f'ymap_{i}'].astype(np.float32))
            self.masks.append(data[f'mask_{i}'])
            self.corners.append(data[f'corner_{i}'].tolist())
            self.sizes.append(data[f'size_{i}'].tolist())

        print(f"  ✅ 워핑 맵 {self.num_cameras}개 로드 완료")

        # GPU 업로드
        if CUPY_AVAILABLE:
            self._upload_gpu()

        return True

    def _upload_gpu(self):
        """워핑 맵을 GPU 메모리에 업로드"""
        print("\n[GPU] 메모리 업로드...")
        self.gpu_xmaps = []
        self.gpu_ymaps = []
        self.gpu_masks = []

        for i in range(self.num_cameras):
            self.gpu_xmaps.append(cp.asarray(self.xmaps[i]))
            self.gpu_ymaps.append(cp.asarray(self.ymaps[i]))
            # 마스크를 float32로 변환 (0~1)
            mask_f = self.masks[i].astype(np.float32) / 255.0
            self.gpu_masks.append(cp.asarray(mask_f))

        mem_used = cp.get_default_memory_pool().used_bytes() / (1024 ** 2)
        print(f"  ✅ GPU 메모리: {mem_used:.1f} MB")

    def warmup(self, num_warmup=3):
        """GPU 워밍업 (첫 실행 지연 방지)"""
        if not CUPY_AVAILABLE:
            return
        print("\n[GPU] 워밍업 중...")
        dummy = [np.random.randint(0, 255, (self.img_h, self.img_w, 3), dtype=np.uint8)
                 for _ in range(self.num_cameras)]
        for _ in range(num_warmup):
            try:
                self.stitch_gpu(dummy)
            except Exception:
                pass
        cp.cuda.Stream.null.synchronize()
        print("  ✅ 워밍업 완료")

    def stitch_gpu(self, images):
        """cv2.remap() 워핑 + CuPy GPU 블렌딩"""
        # 스케일 적용
        if self.scale != 1.0:
            images = [cv.resize(img, None, fx=self.scale, fy=self.scale,
                                interpolation=cv.INTER_AREA) for img in images]

        pano = cp.zeros((self.pano_h, self.pano_w, 3), dtype=cp.float32)
        weight_sum = cp.zeros((self.pano_h, self.pano_w), dtype=cp.float32)

        for i in range(self.num_cameras):
            # cv2.remap()으로 워핑 (CPU, 검증된 함수)
            warped = cv.remap(images[i], self.xmaps[i], self.ymaps[i],
                              cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
            mask = self.masks[i].astype(np.float32) / 255.0

            wh, ww = warped.shape[:2]

            # 파노라마 좌표
            cx, cy = self.corners[i]
            px = cx - self.pano_x
            py = cy - self.pano_y

            # 범위 클리핑
            src_y1 = max(0, -py)
            src_x1 = max(0, -px)
            dst_y1 = max(0, py)
            dst_x1 = max(0, px)

            copy_h = min(wh - src_y1, self.pano_h - dst_y1)
            copy_w = min(ww - src_x1, self.pano_w - dst_x1)

            if copy_h <= 0 or copy_w <= 0:
                continue

            warped_region = warped[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]
            mask_region = mask[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]

            # GPU 블렌딩
            gpu_warped = cp.asarray(warped_region.astype(np.float32))
            gpu_mask = cp.asarray(mask_region)

            pano[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] += \
                gpu_warped * gpu_mask[:, :, cp.newaxis]
            weight_sum[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] += gpu_mask

        # 정규화
        weight_sum = cp.maximum(weight_sum, 1e-6)
        pano = pano / weight_sum[:, :, cp.newaxis]

        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(cp.clip(pano, 0, 255)).astype(np.uint8)

    def stitch_cpu(self, images):
        """CPU remap 폴백"""
        if self.scale != 1.0:
            images = [cv.resize(img, None, fx=self.scale, fy=self.scale,
                                interpolation=cv.INTER_AREA) for img in images]

        result = np.zeros((self.pano_h, self.pano_w, 3), dtype=np.float32)
        weight_sum = np.zeros((self.pano_h, self.pano_w), dtype=np.float32)

        for i in range(self.num_cameras):
            warped = cv.remap(images[i], self.xmaps[i], self.ymaps[i],
                              cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
            mask = self.masks[i].astype(np.float32) / 255.0

            cx, cy = self.corners[i]
            wh, ww = warped.shape[:2]

            px = cx - self.pano_x
            py = cy - self.pano_y

            src_y1 = max(0, -py)
            src_x1 = max(0, -px)
            dst_y1 = max(0, py)
            dst_x1 = max(0, px)

            copy_h = min(wh - src_y1, self.pano_h - dst_y1)
            copy_w = min(ww - src_x1, self.pano_w - dst_x1)

            if copy_h <= 0 or copy_w <= 0:
                continue

            warped_region = warped[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]
            mask_region = mask[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]

            result[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] += \
                warped_region.astype(np.float32) * mask_region[:, :, np.newaxis]
            weight_sum[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] += mask_region

        valid = weight_sum > 0
        result[valid] /= weight_sum[valid, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)

    def stitch(self, images):
        """자동 GPU/CPU 선택"""
        if CUPY_AVAILABLE:
            return self.stitch_gpu(images)
        else:
            return self.stitch_cpu(images)


# ============================================================
# 영상 수집율 시험
# ============================================================
class CollectionRateTest:
    """영상 수집율 시험 평가"""

    def __init__(self, args):
        self.args = args
        self.stitcher = CuPyStitcher()
        self.receivers = []

    def init_receivers(self):
        """UDP 수신기 초기화"""
        print("\n" + "=" * 60)
        print("[UDP] 수신기 초기화")
        print("=" * 60)
        for i, port in enumerate(self.args.ports):
            receiver = UDPReceiver(port, f"UDP_{i + 1}")
            receiver.start()
            self.receivers.append(receiver)
        print("  프레임 수신 대기 중...")
        time.sleep(3)
        print("  ✅ UDP 수신기 준비 완료")

    def stop_receivers(self):
        """UDP 수신기 정지"""
        for r in self.receivers:
            r.stop()
        self.receivers = []

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
            h, w = frame.shape[:2]
            h2, w2 = h // 2, w // 2
            images_8cam.extend([
                frame[0:h2, 0:w2],
                frame[0:h2, w2:w],
                frame[h2:h, 0:w2],
                frame[h2:h, w2:w]
            ])

        # 카메라 순서 재배치
        camera_order = self.args.camera_order
        reordered = [images_8cam[i - 1] for i in camera_order]
        return reordered

    def run_single_trial(self, trial_num, output_dir):
        """단일 시험 실행 (2분)"""
        DURATION = 120  # 2분
        INTERVAL = 1.0  # 1초당 1장
        TARGET_TOTAL = 120  # 목표 120장

        trial_dir = os.path.join(output_dir, f"trial_{trial_num}")
        os.makedirs(trial_dir, exist_ok=True)
        csv_path = os.path.join(trial_dir, f"trial_{trial_num}_result.csv")

        print(f"\n{'=' * 60}")
        print(f"[시험 {trial_num}/{self.args.num_trials}] 시작")
        print(f"{'=' * 60}")
        print(f"  목표: {DURATION}초 동안 {TARGET_TOTAL}장 생성")
        print(f"  간격: {INTERVAL}초")
        print(f"  모드: {'CuPy GPU' if CUPY_AVAILABLE else 'CPU'}")
        print(f"  저장: {trial_dir}")
        print()

        # CSV 헤더
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(csv_file)
        writer.writerow([
            'Frame_Number',
            'Target_Time_s',
            'Actual_Time_s',
            'Generation_Time',
            'Stitch_Time_ms',
            'Total_Time_ms',
            'Success',
            'Image_File'
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

            # 2. GPU 스티칭
            stitch_start = time.time()
            try:
                pano = self.stitcher.stitch(images)
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
                cv.imwrite(img_path, pano, [cv.IMWRITE_JPEG_QUALITY, 85])

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
                rate = (success_count / frame_num) * 100 if frame_num > 0 else 0
                avg_s = total_stitch_time / success_count if success_count > 0 else 0
                print(f"  [{elapsed:.0f}s] 프레임 {frame_num} | "
                      f"성공: {success_count} | 실패: {fail_count} | "
                      f"수집율: {rate:.1f}% | 평균 스티칭: {avg_s:.0f}ms")

            # 다음 캡처 시간
            next_capture += INTERVAL

            # 밀린 프레임 처리 (처리가 1초 이상 걸린 경우)
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
        collection_rate = (success_count / TARGET_TOTAL) * 100 if TARGET_TOTAL > 0 else 0
        avg_stitch = total_stitch_time / success_count if success_count > 0 else 0

        print(f"\n  {'=' * 40}")
        print(f"  [시험 {trial_num}] 결과")
        print(f"  {'=' * 40}")
        print(f"  목표 프레임: {TARGET_TOTAL}장")
        print(f"  시도 프레임: {total_frames}장")
        print(f"  성공 프레임: {success_count}장")
        print(f"  실패 프레임: {fail_count}장")
        print(f"  수집율: {collection_rate:.1f}%")
        print(f"  평균 스티칭 시간: {avg_stitch:.1f} ms")
        if collection_rate >= 90:
            print(f"  ✅ 목표 달성! (90% 이상)")
        else:
            print(f"  ❌ 목표 미달성 (90% 미만)")
        print(f"  CSV: {csv_path}")

        return {
            'trial': trial_num,
            'target_total': TARGET_TOTAL,
            'total_frames': total_frames,
            'success': success_count,
            'fail': fail_count,
            'collection_rate': collection_rate,
            'avg_stitch_ms': avg_stitch,
        }

    def run(self):
        """전체 시험 실행"""
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1. NPZ 로드
        if not self.stitcher.load_npz(self.args.calibration_npz):
            print("\n❌ NPZ 로드 실패!")
            return

        # 2. UDP 수신기 초기화
        self.init_receivers()

        # 3. GPU 워밍업
        self.stitcher.warmup()

        # 4. 시험 실행
        print("\n" + "=" * 60)
        print("영상 수집율 시험 시작")
        print("=" * 60)
        print(f"  시험 횟수: {self.args.num_trials}회")
        print(f"  시험 시간: 120초 (2분)")
        print(f"  목표 수집율: 90%")
        print(f"  목표 프레임: 1초당 1장 (120장/2분)")
        print(f"  모드: {'CuPy GPU' if CUPY_AVAILABLE else 'CPU'}")
        print(f"  스케일: {self.stitcher.scale * 100:.0f}%")
        print(f"  출력: {output_dir}")

        all_results = []

        for trial in range(1, self.args.num_trials + 1):
            if trial > 1:
                wait = self.args.trial_interval
                print(f"\n⏳ 다음 시험까지 {wait}초 대기...")
                time.sleep(wait)

            result = self.run_single_trial(trial, output_dir)
            all_results.append(result)

        # 수신기 정지
        self.stop_receivers()

        # 전체 요약 CSV
        summary_csv = os.path.join(output_dir, "summary.csv")
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'Trial', 'Target_Total', 'Total_Frames', 'Success',
                'Fail', 'Collection_Rate_%', 'Avg_Stitch_ms'
            ])
            for r in all_results:
                w.writerow([
                    r['trial'], r['target_total'], r['total_frames'],
                    r['success'], r['fail'],
                    f"{r['collection_rate']:.1f}",
                    f"{r['avg_stitch_ms']:.1f}"
                ])
            # 평균
            avg_rate = np.mean([r['collection_rate'] for r in all_results])
            avg_stitch = np.mean([r['avg_stitch_ms'] for r in all_results])
            w.writerow([])
            w.writerow(['Average', '', '', '', '',
                         f"{avg_rate:.1f}", f"{avg_stitch:.1f}"])

        # 최종 결과 출력
        print("\n" + "=" * 60)
        print("전체 시험 결과 요약")
        print("=" * 60)
        print(f"{'시험':>6} | {'목표':>6} | {'성공':>6} | {'실패':>6} | "
              f"{'수집율':>8} | {'스티칭':>10}")
        print("-" * 60)
        for r in all_results:
            status = "✅" if r['collection_rate'] >= 90 else "❌"
            print(f"  {r['trial']:>4} | {r['target_total']:>5}장 | "
                  f"{r['success']:>5}장 | {r['fail']:>5}장 | "
                  f"{r['collection_rate']:>6.1f}% {status} | "
                  f"{r['avg_stitch_ms']:>8.1f}ms")
        print("-" * 60)
        avg_rate = np.mean([r['collection_rate'] for r in all_results])
        avg_stitch = np.mean([r['avg_stitch_ms'] for r in all_results])
        overall = "✅" if avg_rate >= 90 else "❌"
        print(f"  평균 |       |       |       | "
              f"{avg_rate:>6.1f}% {overall} | {avg_stitch:>8.1f}ms")
        print("=" * 60)
        print(f"\n✅ 요약 CSV: {summary_csv}")
        print(f"✅ 각 시험 결과: {output_dir}/trial_N/trial_N_result.csv")


def main():
    parser = argparse.ArgumentParser(
        description="GPU Step 2: CuPy GPU 스티칭 + 영상 수집율 시험")

    parser.add_argument("--calibration_npz", type=str, required=True,
                        help="캘리브레이션 NPZ 파일 (gpu_step1_calibrate.py로 생성)")
    parser.add_argument("--ports", type=int, nargs="+", default=[5001, 5002],
                        help="UDP 포트 (기본: 5001 5002)")
    parser.add_argument("--camera_order", type=int, nargs="+",
                        default=[5, 4, 3, 2, 1, 8, 7, 6],
                        help="카메라 순서")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="시험 횟수 (기본: 5)")
    parser.add_argument("--trial_interval", type=int, default=10,
                        help="시험 간 대기 시간 초 (기본: 10)")
    parser.add_argument("--output_dir", type=str,
                        default="./output/collection_test",
                        help="결과 저장 디렉토리")

    args = parser.parse_args()

    print("=" * 60)
    print("영상 수집율 시험 평가 (CuPy GPU 가속)")
    print("=" * 60)
    print(f"캘리브레이션 NPZ: {args.calibration_npz}")
    print(f"UDP 포트: {args.ports}")
    print(f"카메라 순서: {args.camera_order}")
    print(f"시험 횟수: {args.num_trials}회")
    print(f"시험 시간: 120초 (2분)")
    print(f"목표: 1초당 1장 (120장/2분)")
    print(f"목표 수집율: 90%")
    print(f"GPU: {'CuPy' if CUPY_AVAILABLE else 'CPU'}")
    print(f"출력: {args.output_dir}")
    print("=" * 60)

    tester = CollectionRateTest(args)
    tester.run()


if __name__ == "__main__":
    main()
