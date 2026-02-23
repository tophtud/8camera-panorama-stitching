#!/usr/bin/env python3
"""
GPU Step 1: cv2.detail API 기반 캘리브레이션
- OpenCV detail 모듈로 특징점 매칭, 카메라 추정, 번들 조정
- SphericalWarper로 워핑 맵(xmap, ymap) 추출
- NPZ 파일로 저장 (카메라 파라미터 + 워핑 맵)
"""

import os
import glob
import argparse
import numpy as np
import cv2 as cv


def load_camera_images(input_dir, num_frames=10, camera_order=None):
    """폴더에서 이미지 로드"""
    print("\n[폴더] 캘리브레이션 이미지 로드")
    
    if camera_order is None:
        camera_order = list(range(1, 9))
    
    print(f"  카메라 순서: {camera_order}")
    
    all_images = []
    
    for cam_idx in camera_order:
        cam_dir = os.path.join(input_dir, f'MyCam_{cam_idx:03d}')
        
        if not os.path.exists(cam_dir):
            print(f"  ❌ 카메라 {cam_idx} 디렉토리 없음")
            continue
        
        img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[:num_frames]
        images = [cv.imread(f) for f in img_files if cv.imread(f) is not None]
        
        if images:
            all_images.append(images)
            print(f"  ✅ 카메라 {cam_idx}: {len(images)}장")
    
    print(f"  총 {len(all_images)}개 카메라\n")
    return all_images


def calibrate(ref_images, scale=1.0, conf_thresh=0.5):
    """cv2.detail API로 캘리브레이션"""
    
    num_images = len(ref_images)
    
    # 스케일 적용
    if scale != 1.0:
        images = [cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) for img in ref_images]
    else:
        images = ref_images
    
    img_sizes = [(img.shape[1], img.shape[0]) for img in images]
    h, w = images[0].shape[:2]
    
    print(f"  입력: {num_images}개 카메라")
    print(f"  크기: {w} x {h}")
    
    # ========================================
    # 1. 특징점 검출
    # ========================================
    print("\n[1/6] 특징점 검출 (SIFT)")
    finder = cv.SIFT_create()
    features = []
    for i, img in enumerate(images):
        feat = cv.detail.computeImageFeatures2(finder, img)
        features.append(feat)
        print(f"  카메라 {i+1}: {len(feat.keypoints)}개 특징점")
    
    # ========================================
    # 2. 특징점 매칭
    # ========================================
    print("\n[2/6] 특징점 매칭")
    matcher = cv.detail_BestOf2NearestMatcher(False, 0.65)
    matches_info = matcher.apply2(features)
    matcher.collectGarbage()
    
    # 매칭 결과 출력
    for m in matches_info:
        if m.src_img_idx >= 0 and m.dst_img_idx >= 0 and m.confidence > 0:
            print(f"  카메라 {m.src_img_idx+1} ↔ {m.dst_img_idx+1}: "
                  f"confidence={m.confidence:.3f}, inliers={m.num_inliers}")
    
    # ========================================
    # 3. 카메라 파라미터 추정
    # ========================================
    print("\n[3/6] 카메라 파라미터 추정")
    estimator = cv.detail_HomographyBasedEstimator()
    b, cameras = estimator.apply(features, matches_info, None)
    if not b:
        print("  ❌ 카메라 파라미터 추정 실패!")
        return None
    
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
    
    print(f"  ✅ {len(cameras)}개 카메라 파라미터 추정 성공!")
    
    # ========================================
    # 4. 번들 조정
    # ========================================
    print("\n[4/6] 번들 조정 (BundleAdjusterRay)")
    adjuster = cv.detail_BundleAdjusterRay()
    adjuster.setConfThresh(conf_thresh)
    refine_mask = np.zeros((3, 3), np.uint8)
    refine_mask[0, 0] = 1  # fx
    refine_mask[0, 1] = 1  # skew
    refine_mask[0, 2] = 1  # ppx
    refine_mask[1, 1] = 1  # aspect
    refine_mask[1, 2] = 1  # ppy
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, matches_info, cameras)
    if not b:
        print("  ❌ 번들 조정 실패!")
        return None
    
    print("  ✅ 번들 조정 성공!")
    
    # 초점 거리 계산
    focals = sorted([cam.focal for cam in cameras])
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
    
    print(f"  워핑 스케일: {warped_image_scale:.2f}")
    
    # 수평 보정
    rmats = [np.copy(cam.R) for cam in cameras]
    rmats = cv.detail.waveCorrect(rmats, cv.detail.WAVE_CORRECT_HORIZ)
    for idx, cam in enumerate(cameras):
        cam.R = rmats[idx]
    
    # ========================================
    # 5. 워핑 맵 생성 (SphericalWarper)
    # ========================================
    print("\n[5/6] 워핑 맵 생성 (SphericalWarper)")
    warper = cv.PyRotationWarper('spherical', warped_image_scale)
    
    corners = []
    sizes = []
    xmaps = []
    ymaps = []
    masks_warped = []
    K_matrices = []
    R_matrices = []
    
    for idx in range(num_images):
        K = cameras[idx].K().astype(np.float32)
        R = cameras[idx].R.astype(np.float32)
        
        K_matrices.append(K)
        R_matrices.append(R)
        
        # buildMaps로 워핑 맵 추출
        src_size = (images[idx].shape[1], images[idx].shape[0])
        rect_or_corner, xmap, ymap = warper.buildMaps(src_size, K, R)
        
        # buildMaps는 Rect(x,y,w,h) 또는 Point(x,y)를 반환
        if len(rect_or_corner) == 4:
            # Rect (x, y, w, h) 형식
            cx, cy = int(rect_or_corner[0]), int(rect_or_corner[1])
        elif len(rect_or_corner) == 2:
            cx, cy = int(rect_or_corner[0]), int(rect_or_corner[1])
        else:
            cx, cy = int(rect_or_corner[0]), int(rect_or_corner[1])
        
        corners.append((cx, cy))
        sizes.append((xmap.shape[1], xmap.shape[0]))
        xmaps.append(xmap)
        ymaps.append(ymap)
        
        # 마스크 워핑
        mask = 255 * np.ones((images[idx].shape[0], images[idx].shape[1]), np.uint8)
        _, mask_warped = warper.warp(mask, K, R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_warped)
        
        print(f"  카메라 {idx+1}: corner=({cx},{cy}), size={sizes[-1]}")
    
    # 파노라마 크기 계산 (수동)
    all_x = [c[0] for c in corners]
    all_y = [c[1] for c in corners]
    all_x2 = [c[0] + s[0] for c, s in zip(corners, sizes)]
    all_y2 = [c[1] + s[1] for c, s in zip(corners, sizes)]
    
    pano_x = min(all_x)
    pano_y = min(all_y)
    pano_w = max(all_x2) - pano_x
    pano_h = max(all_y2) - pano_y
    dst_sz = (pano_x, pano_y, pano_w, pano_h)
    print(f"\n  파노라마: {pano_w}x{pano_h} (offset: {pano_x},{pano_y})")
    
    # ========================================
    # 6. 기준 파노라마 생성 (검증용)
    # ========================================
    print("\n[6/6] 기준 파노라마 생성")
    
    # 블렌더 생성
    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
    blend_width = np.sqrt(pano_w * pano_h) * 5 / 100
    if blend_width > 1:
        blender = cv.detail_MultiBandBlender()
        blender.setNumBands(int((np.log(blend_width) / np.log(2.) - 1.)))
    blender.prepare((int(pano_x), int(pano_y), int(pano_w), int(pano_h)))
    
    for idx in range(num_images):
        K = K_matrices[idx]
        R = R_matrices[idx]
        
        # 워핑
        corner, image_warped = warper.warp(images[idx], K, R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        mask = 255 * np.ones((images[idx].shape[0], images[idx].shape[1]), np.uint8)
        _, mask_warped = warper.warp(mask, K, R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        
        image_warped_s = image_warped.astype(np.int16)
        feed_corner = (int(corners[idx][0]), int(corners[idx][1]))
        blender.feed(cv.UMat(image_warped_s), mask_warped, feed_corner)
    
    result, result_mask = blender.blend(None, None)
    panorama = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    
    print(f"  ✅ 파노라마: {panorama.shape[1]}x{panorama.shape[0]}")
    
    return {
        'cameras': cameras,
        'K_matrices': K_matrices,
        'R_matrices': R_matrices,
        'warped_image_scale': warped_image_scale,
        'corners': corners,
        'sizes': sizes,
        'xmaps': xmaps,
        'ymaps': ymaps,
        'masks_warped': masks_warped,
        'pano_rect': dst_sz,
        'panorama': panorama,
        'img_size': (w, h)
    }


def save_npz(result, args):
    """NPZ 파일로 저장"""
    print("\n[저장] NPZ 파일 생성")
    
    pano_x, pano_y, pano_w, pano_h = result['pano_rect']
    
    save_dict = {
        'scale': args.scale,
        'camera_order': np.array(args.camera_order if args.camera_order else list(range(1, 9))),
        'image_width': result['img_size'][0],
        'image_height': result['img_size'][1],
        'warped_image_scale': result['warped_image_scale'],
        'pano_x': pano_x,
        'pano_y': pano_y,
        'pano_width': pano_w,
        'pano_height': pano_h,
        'num_cameras': len(result['cameras']),
    }
    
    # 카메라별 데이터 저장
    for i in range(len(result['cameras'])):
        save_dict[f'K_{i}'] = result['K_matrices'][i]
        save_dict[f'R_{i}'] = result['R_matrices'][i]
        save_dict[f'corner_{i}'] = np.array(result['corners'][i], dtype=np.int32)
        save_dict[f'size_{i}'] = np.array(result['sizes'][i], dtype=np.int32)
        save_dict[f'xmap_{i}'] = result['xmaps'][i]
        save_dict[f'ymap_{i}'] = result['ymaps'][i]
        save_dict[f'mask_{i}'] = result['masks_warped'][i]
    
    np.savez_compressed(args.output_npz, **save_dict)
    
    file_size = os.path.getsize(args.output_npz) / (1024 * 1024)
    print(f"  ✅ NPZ 저장: {args.output_npz} ({file_size:.1f} MB)")
    print(f"  카메라: {len(result['cameras'])}개")
    print(f"  파노라마: {pano_w}x{pano_h}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="GPU Step 1: cv2.detail API 캘리브레이션")
    
    parser.add_argument("--calibration_dir", type=str, required=True, help="캘리브레이션 이미지 폴더")
    parser.add_argument("--calibration_frames", type=int, default=10)
    parser.add_argument("--reference_frame", type=int, default=7)
    parser.add_argument("--camera_order", type=int, nargs="+", default=None)
    parser.add_argument("--scale", type=float, default=1.0, help="이미지 스케일")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="신뢰도 임계값")
    parser.add_argument("--output_npz", type=str, default="calibration_gpu.npz", help="출력 NPZ 파일")
    parser.add_argument("--save_reference", type=str, default="reference_gpu.jpg", help="기준 파노라마 이미지")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("파노라마 스티칭 - GPU Step 1: 캘리브레이션")
    print("=" * 60)
    print(f"캘리브레이션 폴더: {args.calibration_dir}")
    print(f"기준 프레임: {args.reference_frame}")
    print(f"스케일: {args.scale * 100:.0f}%")
    if args.camera_order:
        print(f"카메라 순서: {args.camera_order}")
    print(f"출력 NPZ: {args.output_npz}")
    print("=" * 60)
    
    # 이미지 로드
    all_images = load_camera_images(
        os.path.expanduser(args.calibration_dir),
        num_frames=args.calibration_frames,
        camera_order=args.camera_order
    )
    
    if len(all_images) < 2:
        print("❌ 카메라 부족")
        return
    
    # 기준 프레임
    ref_idx = min(args.reference_frame - 1, len(all_images[0]) - 1)
    ref_images = [cam_images[ref_idx] for cam_images in all_images]
    
    print(f"[기준] 프레임 {ref_idx+1}/{len(all_images[0])}")
    
    # 캘리브레이션
    print("\n" + "=" * 60)
    print("캘리브레이션 시작")
    print("=" * 60)
    
    result = calibrate(ref_images, scale=args.scale, conf_thresh=args.conf_thresh)
    
    if result is None:
        print("\n❌ 캘리브레이션 실패!")
        return
    
    # NPZ 저장
    save_npz(result, args)
    
    # 기준 파노라마 저장
    if args.save_reference:
        cv.imwrite(args.save_reference, result['panorama'])
        print(f"  ✅ 기준 파노라마 저장: {args.save_reference}")
    
    print("\n✅ 캘리브레이션 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print(f"  python3 gpu_step2_realtime.py --calibration_npz {args.output_npz} --ports 5001 5002")
    print("=" * 60)


if __name__ == "__main__":
    main()
