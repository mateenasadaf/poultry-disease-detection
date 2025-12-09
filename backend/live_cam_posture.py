# live_disease_cam_posture.py
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import joblib
import math
from collections import deque, Counter

# ---------- PATHS (update if needed) ----------
YOLO_CHICKEN_PATH = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\chicken_detector.pt"
YOLO_POSE_PATH    = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\yolov8n-pose.pt"
XCEPTION_PATH     = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\xception_disease.h5"
DISEASE_LABELS    = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\disease_classes.txt"
POSTURE_SCALER    = "posture_scaler.pkl"
POSTURE_MODEL     = "posture_mlp_adv.h5"

# ---------- LOAD MODELS ----------
yolo_chicken = YOLO(YOLO_CHICKEN_PATH)
yolo_chicken.overrides['classes'] = [0]

yolo_pose = YOLO(YOLO_POSE_PATH)

disease_model = load_model(XCEPTION_PATH)
with open(DISEASE_LABELS, "r") as f:
    disease_labels = [line.strip() for line in f.readlines()]

posture_scaler = joblib.load(POSTURE_SCALER)
posture_model = load_model(POSTURE_MODEL)
posture_labels = ["healthy", "sick"]

# ---------- UTILS ----------
def angle_between(pA, pB, pC):
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
    try:
        p = kpts[idx]
        if np.isnan(p).any():
            return (float("nan"), float("nan"))
        return (float(p[0]), float(p[1]))
    except Exception:
        return (float("nan"), float("nan"))

# same KP mapping as extractor
KP = {
    "nose":0, "l_eye":1, "r_eye":2, "l_ear":3, "r_ear":4,
    "l_shoulder":5, "r_shoulder":6, "l_elbow":7, "r_elbow":8,
    "l_wrist":9, "r_wrist":10, "l_hip":11, "r_hip":12,
    "l_knee":13, "r_knee":14, "l_ankle":15, "r_ankle":16
}

def compute_features_from_pose(kpts, box):
    """Compute the same feature vector as in extractor. kpts are in crop coords (Nx2). box is xyxy int in crop coords"""
    x1,y1,x2,y2 = box
    bw = max(1, x2-x1); bh = max(1, y2-y1)
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

    mid_sh = ((l_sh[0]+r_sh[0])/2.0, (l_sh[1]+r_sh[1])/2.0)
    mid_hip = ((l_hip[0]+r_hip[0])/2.0, (l_hip[1]+r_hip[1])/2.0)
    tail_approx = (mid_hip[0] + (mid_hip[0]-mid_sh[0]), mid_hip[1] + (mid_hip[1]-mid_sh[1]))

    neck_angle = angle_between(nose, mid_sh, mid_hip)
    back_angle = angle_between(mid_sh, mid_hip, tail_approx)
    left_wing_angle = angle_between(l_sh, l_elb, l_wri)
    right_wing_angle = angle_between(r_sh, r_elb, r_wri)
    left_leg_angle = angle_between(l_hip, l_knee, l_ank)
    right_leg_angle = angle_between(r_hip, r_knee, r_ank)

    beak_to_hip = dist(nose, mid_hip)
    body_len = dist(mid_sh, mid_hip)
    left_leg_len = dist(l_hip, l_ank)
    right_leg_len = dist(r_hip, r_ank)

    leg_ratio = float("nan")
    if right_leg_len and not math.isnan(right_leg_len) and right_leg_len != 0:
        leg_ratio = left_leg_len / right_leg_len

    abs_wing_diff = abs(left_wing_angle - right_wing_angle) if not (math.isnan(left_wing_angle) or math.isnan(right_wing_angle)) else float("nan")
    abs_leg_diff = abs(left_leg_angle - right_leg_angle) if not (math.isnan(left_leg_angle) or math.isnan(right_leg_angle)) else float("nan")

    head_y_rel = (nose[1] - y1) / bh if not math.isnan(nose[1]) else float("nan")
    com_x = np.nanmean([mid_sh[0], mid_hip[0]]) if not (math.isnan(mid_sh[0]) and math.isnan(mid_hip[0])) else float("nan")
    center_of_mass_x_rel = (com_x - x1) / bw if not math.isnan(com_x) else float("nan")

    kpts_arr = np.array(kpts)
    num_valid_kpts = int(np.sum(~np.isnan(kpts_arr[:,0])))

    feats = [
        neck_angle, back_angle,
        left_wing_angle, right_wing_angle,
        left_leg_angle, right_leg_angle,
        beak_to_hip, body_len, left_leg_len, right_leg_len, leg_ratio,
        abs_wing_diff, abs_leg_diff,
        head_y_rel, center_of_mass_x_rel,
        num_valid_kpts
    ]
    return np.array(feats, dtype=np.float32)

# smoothing helper
def majority_vote(deq, fallback="unknown"):
    if not deq:
        return fallback
    cnt = Counter(deq)
    return cnt.most_common(1)[0][0]

# ---------- START WEBCAM ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    raise SystemExit

SMOOTH_LEN = 7

print("✔ Webcam started — running disease + advanced posture")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_chicken(frame, verbose=False)[0]
        posture_deques = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.75:
                continue
            cls = int(box.cls[0])
            if cls != 0:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            w = x2-x1; h = y2-y1
            if w < 80 or h < 80:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            # disease prediction (same as before)
            try:
                c_rt = cv2.resize(crop, (299,299))
                c_norm = c_rt.astype("float32")/255.0
                pred = disease_model.predict(np.expand_dims(c_norm,0), verbose=0)
                d_idx = int(np.argmax(pred))
                disease_name = disease_labels[d_idx] if d_idx < len(disease_labels) else "unknown"
                disease_conf = float(pred[0][d_idx])
            except Exception:
                disease_name = "d_err"
                disease_conf = 0.0

            # pose on crop
            try:
                res_pose = yolo_pose(crop, verbose=False)[0]
                if len(res_pose.keypoints) > 0 and len(res_pose.boxes) > 0:
                    kpts = res_pose.keypoints.xy[0].cpu().numpy()
                    # pad if smaller
                    if kpts.shape[0] < 17:
                        pad = np.full((17-kpts.shape[0],2), np.nan)
                        kpts = np.vstack([kpts, pad])

                    # box for pose detection (in crop coords)
                    box_pose = res_pose.boxes[0].xyxy[0].cpu().numpy().astype(int)
                    feats = compute_features_from_pose(kpts, box_pose)

                    # impute NaN with column medians (simple runtime impute)
                    nan_inds = np.isnan(feats)
                    if nan_inds.any():
                        # for single sample, replace NaN with 0 (after scaler it'll be handled); better: use median from training saved scaler mean_ maybe not available
                        feats[nan_inds] = 0.0

                    feats_scaled = posture_scaler.transform(feats.reshape(1, -1))
                    p_pred = posture_model.predict(feats_scaled, verbose=0)
                    p_idx = int(np.argmax(p_pred))
                    p_label = posture_labels[p_idx]
                else:
                    p_label = "no_kp"
            except Exception as e:
                p_label = "pose_err"

            # smoothing: maintain a short deque per detection on-the-fly using list of deques
            # Here we simply create a deque per detection in this frame (no cross-frame identity).
            # For short-term smoothing across frames, you can maintain a global list mapping boxes->deques via IoU tracking.
            dq = deque(maxlen=SMOOTH_LEN)
            dq.append(p_label)
            smoothed = majority_vote(dq, fallback=p_label)

            label = f"{disease_name} ({disease_conf:.2f}) | Posture: {smoothed}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), (0,200,0), -1)
            cv2.putText(frame, label, (x1+3, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow("Live Disease + Advanced Posture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✔ Stopped")
