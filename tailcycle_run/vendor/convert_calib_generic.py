"""Convert an OpenCV ArUco-style calibration dir to JARVIS format.

Usage: python convert_calib_generic.py <src_dir> <dst_dir>

Source dir must contain CamXXXXXXX.yaml files with OpenCV-format keys:
  camera_matrix, distortion_coefficients, rc_ext, tc_ext.

Output dir gets per-cam YAMLs with JARVIS keys:
  intrinsicMatrix (K^T), distortionCoefficients (1x5), R (R^T), T (3x1).
"""
import os, sys, cv2

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)

srcs = sorted(f for f in os.listdir(src) if f.startswith("Cam") and f.endswith(".yaml"))
if not srcs:
    print(f"No Cam*.yaml in {src}")
    sys.exit(2)

for fn in srcs:
    fs = cv2.FileStorage(os.path.join(src, fn), cv2.FILE_STORAGE_READ)
    K, D = fs.getNode("camera_matrix").mat(), fs.getNode("distortion_coefficients").mat()
    R, T = fs.getNode("rc_ext").mat(), fs.getNode("tc_ext").mat()
    fs.release()
    if any(x is None for x in (K, D, R, T)):
        print(f"  SKIP {fn} (missing fields)"); continue
    fw = cv2.FileStorage(os.path.join(dst, fn), cv2.FILE_STORAGE_WRITE)
    fw.write("intrinsicMatrix", K.T)
    fw.write("distortionCoefficients", D.reshape(1, -1))
    fw.write("R", R.T)
    fw.write("T", T.reshape(3, 1))
    fw.release()
print(f"converted {len(srcs)} cams from {src} -> {dst}")
