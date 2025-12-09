# extract_posture_features_adv.py
import os
import numpy as np
from ultralytics import YOLO
import cv2
import math
from pathlib import Path

# ---------- CONFIGURE PATHS ----------
POSE_MODEL = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\yolov8n-pose.pt"
DATASET = r"C:\Users\mateena sadaf\Desktop\poultry disease\datasets\posture_dataset"  # folders: healthy/ sick/
OUT_NPZ = "posture_adv_features.npz"
OUT_FEATURE_NAMES = "feature_names.txt"

# ---------- UTIL FUNCTIONS ----------
def angle_between(pA, pB, pC):
    """Angle at pB formed by pA-pB-pC in degrees. Points are (x,y). Returns NaN if invalid."""
    try:
        BA = (pA[0]-pB[0], pA[1]-pB[1])
        BC = (pC[0]-pB[0], pC[1]-pB[1])
        dot = BA[0]*BC[0] + BA[1]*BC[1]
        magA = math.hypot(BA[0], BA[1])
        magC = math.hypot(BC[0], BC[1])
        if magA == 0 or magC == 0:
            return float("nan")
        cosang = max(-1.0, min(1.0, dot/(magA*magC)))
        return math.degrees(math.acos(cosang))
    except Exception:
        return float("nan")

def dist(pA, pB):
    try:
        return math.hypot(pA[0]-pB[0], pA[1]-pB[1])
    except Exception:
        return float("nan")

def safe_get(kpts, idx):
    """Return (x,y) or (nan,nan) if index missing or invalid."""
    try:
        p = kpts[idx]
        if np.isnan(p).any():
            return (float("nan"), float("nan"))
        return (float(p[0]), float(p[1]))
    except Exception:
        return (float("nan"), float("nan"))

# ---------- LOAD MODEL ----------
model = YOLO(POSE_MODEL)

# ---------- KEYPOINT INDEX MAP (COCO-style assumed) ----------
# If your pose model uses different indices, change these values.
KP = {
    "nose":0, "l_eye":1, "r_eye":2, "l_ear":3, "r_ear":4,
    "l_shoulder":5, "r_shoulder":6, "l_elbow":7, "r_elbow":8,
    "l_wrist":9, "r_wrist":10, "l_hip":11, "r_hip":12,
    "l_knee":13, "r_knee":14, "l_ankle":15, "r_ankle":16
}
EXPECTED_K = 17

# ---------- FEATURE LIST (names kept for saving) ----------
feature_names = [
    # Angles
    "neck_angle_nose_shoulder_mid",       # nose - mid_shoulder - mid_hip
    "back_angle_shouldermid_hipmid_tail", # mid_shoulder - mid_hip - hip midpoint->approx tail (use mid_hip+vector)
    "left_wing_angle_shoulder_elbow_wrist",
    "right_wing_angle_shoulder_elbow_wrist",
    "left_leg_angle_hip_knee_ankle",
    "right_leg_angle_hip_knee_ankle",
    # Distances & ratios
    "beak_to_hip_dist",
    "body_length_shouldermid_hipmid",
    "left_leg_length",
    "right_leg_length",
    "leg_length_ratio",                    # left/right ratio
    # Symmetry (abs diffs)
    "abs_wing_angle_diff",
    "abs_leg_angle_diff",
    # Relative positions (normalized)
    "head_y_rel",                          # head y relative to bbox height
    "center_of_mass_x_rel",                # x of mean of shoulders+hips relative to bbox
    # Keypoint completeness
    "num_valid_kpts",
]

# ---------- PROCESS DATA ----------
X = []
y = []
labels = ["healthy", "sick"]

for label_idx, label_name in enumerate(labels):
    folder = Path(DATASET) / label_name
    if not folder.exists():
        print(f"⚠ Folder not found: {folder}")
        continue
    for imgname in os.listdir(folder):
        if not imgname.lower().endswith((".jpg",".jpeg",".png")):
            continue
        img_path = str(folder / imgname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        res = model(img, verbose=False)[0]
        # require at least one pose detection
        if len(res.keypoints) == 0 or len(res.boxes) == 0:
            continue

        # use first pose detection
        kpts = res.keypoints.xy[0].cpu().numpy()
        # ensure shape
        if kpts.shape[0] < EXPECTED_K:
            # pad with nans to expected length (so indexing safe)
            pad_count = EXPECTED_K - kpts.shape[0]
            kpts = np.vstack([kpts, np.full((pad_count,2), np.nan)])

        # box of pose relative to original image
        box = res.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1,y1,x2,y2 = box
        bw = max(1, x2-x1)
        bh = max(1, y2-y1)

        # get points
        nose = safe_get(kpts, KP["nose"])
        l_sh = safe_get(kpts, KP["l_shoulder"])
        r_sh = safe_get(kpts, KP["r_shoulder"])
        l_hip = safe_get(kpts, KP["l_hip"])
        r_hip = safe_get(kpts, KP["r_hip"])
        l_elb = safe_get(kpts, KP["l_elbow"])
        r_elb = safe_get(kpts, KP["r_elbow"])
        l_wri = safe_get(kpts, KP["l_wrist"])
        r_wri = safe_get(kpts, KP["r_wrist"])
        l_knee = safe_get(kpts, KP["l_knee"])
        r_knee = safe_get(kpts, KP["r_knee"])
        l_ank = safe_get(kpts, KP["l_ankle"])
        r_ank = safe_get(kpts, KP["r_ankle"])

        # mid points
        mid_sh = ((l_sh[0]+r_sh[0])/2.0, (l_sh[1]+r_sh[1])/2.0)
        mid_hip = ((l_hip[0]+r_hip[0])/2.0, (l_hip[1]+r_hip[1])/2.0)

        # compute angles
        neck_angle = angle_between(nose, mid_sh, mid_hip)
        # back angle use shouldermid - hipmid - (hipmid -> hipmid + direction) approximate tail direction by hip->ankle vector average
        # approximate "tail" as mid_hip + (mid_hip - mid_sh) => extend the hip direction for angle
        tail_approx = (mid_hip[0] + (mid_hip[0]-mid_sh[0]), mid_hip[1] + (mid_hip[1]-mid_sh[1]))
        back_angle = angle_between(mid_sh, mid_hip, tail_approx)
        left_wing_angle = angle_between(l_sh, l_elb, l_wri)
        right_wing_angle = angle_between(r_sh, r_elb, r_wri)
        left_leg_angle = angle_between(l_hip, l_knee, l_ank)
        right_leg_angle = angle_between(r_hip, r_knee, r_ank)

        # distances and ratios
        beak_to_hip = dist(nose, mid_hip)
        body_len = dist(mid_sh, mid_hip)
        left_leg_len = dist(l_hip, l_ank)
        right_leg_len = dist(r_hip, r_ank)
        leg_ratio = float("nan")
        if right_leg_len and not math.isnan(right_leg_len) and right_leg_len != 0:
            leg_ratio = left_leg_len / right_leg_len

        # symmetry
        abs_wing_diff = abs(left_wing_angle - right_wing_angle) if not (math.isnan(left_wing_angle) or math.isnan(right_wing_angle)) else float("nan")
        abs_leg_diff = abs(left_leg_angle - right_leg_angle) if not (math.isnan(left_leg_angle) or math.isnan(right_leg_angle)) else float("nan")

        # relative positions (normalize by bbox)
        head_y_rel = (nose[1] - y1) / bh if not math.isnan(nose[1]) else float("nan")
        com_x = np.nanmean([mid_sh[0], mid_hip[0]]) if not (math.isnan(mid_sh[0]) and math.isnan(mid_hip[0])) else float("nan")
        center_of_mass_x_rel = (com_x - x1) / bw if not math.isnan(com_x) else float("nan")

        # count valid kpts
        num_valid_kpts = int(np.sum(~np.isnan(kpts[:,0])))

        feats = [
            neck_angle, back_angle,
            left_wing_angle, right_wing_angle,
            left_leg_angle, right_leg_angle,
            beak_to_hip, body_len, left_leg_len, right_leg_len, leg_ratio,
            abs_wing_diff, abs_leg_diff,
            head_y_rel, center_of_mass_x_rel,
            num_valid_kpts
        ]

        X.append(feats)
        y.append(label_idx)

# convert to numpy (convert NaNs to np.nan OK)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# save
np.savez(OUT_NPZ, X=X, y=y)
with open(OUT_FEATURE_NAMES, "w") as f:
    for name in feature_names:
        f.write(name + "\n")

print(f"✔ Saved features: {OUT_NPZ} (shape X={X.shape}, y={y.shape})")
print(f"✔ Saved feature names: {OUT_FEATURE_NAMES}")
