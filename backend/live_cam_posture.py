# live_disease_cam_posture.py
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import joblib
import math
from collections import deque, Counter

# ---------- PATHS ----------
YOLO_CHICKEN_PATH = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\chicken_detector.pt"
YOLO_POSE_PATH    = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\yolov8n-pose.pt"
XCEPTION_PATH     = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\xception_disease.h5"
DISEASE_LABELS    = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\disease_classes.txt"
POSTURE_SCALER    = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\posture_scaler.pkl"
POSTURE_MODEL     = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\posture_mlp_adv.h5"

# ---------- LOAD MODELS ----------
yolo_chicken = YOLO(YOLO_CHICKEN_PATH)
yolo_chicken.overrides['classes'] = [0]  # detect only chicken class

yolo_pose = YOLO(YOLO_POSE_PATH)

disease_model = load_model(XCEPTION_PATH)
with open(DISEASE_LABELS, "r") as f:
    disease_labels = [line.strip() for line in f.readlines()]

posture_scaler = joblib.load(POSTURE_SCALER)
posture_model = load_model(POSTURE_MODEL)
posture_labels = ["healthy", "sick"]

# ---------- HEALTH FUSION ----------
def fuse_health_status(disease_label, posture_label):
    disease_is_healthy = disease_label.lower() == "healthy"
    posture_is_healthy = posture_label.lower() == "healthy"

    if disease_is_healthy and posture_is_healthy:
        return "Healthy", "🟢 LOW RISK"
    if (not disease_is_healthy) and posture_is_healthy:
        return "Early Stage Disease", "🟡 MODERATE RISK"
    if disease_is_healthy and (not posture_is_healthy):
        return "Posture Problem", "🟠 CAUTION"
    if (not disease_is_healthy) and (not posture_is_healthy):
        return "Severe Health Issue", "🔴 HIGH RISK"

    return "Unknown", "⚪ UNKNOWN"

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
        c = max(-1.0, min(1.0, dot/(magA*magC)))
        return math.degrees(math.acos(c))
    except:
        return float("nan")

def dist(pA, pB):
    try:
        return math.hypot(pA[0]-pB[0], pA[1]-pB[1])
    except:
        return float("nan")

def safe_get(kpts, idx):
    try:
        p = kpts[idx]
        if np.isnan(p).any():
            return (float("nan"), float("nan"))
        return (float(p[0]), float(p[1]))
    except:
        return (float("nan"), float("nan"))

KP = {
    "nose":0, "l_shoulder":5, "r_shoulder":6, "l_elbow":7, "r_elbow":8,
    "l_wrist":9, "r_wrist":10, "l_hip":11, "r_hip":12,
    "l_knee":13, "r_knee":14, "l_ankle":15, "r_ankle":16
}

def compute_features_from_pose(kpts, box):
    x1,y1,x2,y2 = box
    bw = max(1, x2-x1)
    bh = max(1, y2-y1)

    # get kpts
    nose = safe_get(kpts, KP["nose"])
    l_sh = safe_get(kpts, KP["l_shoulder"]); r_sh = safe_get(kpts, KP["r_shoulder"])
    l_hip = safe_get(kpts, KP["l_hip"]); r_hip = safe_get(kpts, KP["r_hip"])
    l_elb = safe_get(kpts, KP["l_elbow"]); r_elb = safe_get(kpts, KP["r_elbow"])
    l_wri = safe_get(kpts, KP["l_wrist"]); r_wri = safe_get(kpts, KP["r_wrist"])
    l_knee = safe_get(kpts, KP["l_knee"]); r_knee = safe_get(kpts, KP["r_knee"])
    l_ank = safe_get(kpts, KP["l_ankle"]); r_ank = safe_get(kpts, KP["r_ankle"])

    mid_sh = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
    mid_hip = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
    tail_apx = (mid_hip[0] + (mid_hip[0]-mid_sh[0]), mid_hip[1] + (mid_hip[1]-mid_sh[1]))

    neck_angle = angle_between(nose, mid_sh, mid_hip)
    back_angle = angle_between(mid_sh, mid_hip, tail_apx)
    lw_angle = angle_between(l_sh, l_elb, l_wri)
    rw_angle = angle_between(r_sh, r_elb, r_wri)
    ll_angle = angle_between(l_hip, l_knee, l_ank)
    rl_angle = angle_between(r_hip, r_knee, r_ank)

    beak_to_hip = dist(nose, mid_hip)
    body_len = dist(mid_sh, mid_hip)
    left_leg_len = dist(l_hip, l_ank)
    right_leg_len = dist(r_hip, r_ank)

    leg_ratio = left_leg_len / right_leg_len if right_leg_len != 0 else float("nan")
    abs_wing_diff = abs(lw_angle - rw_angle)
    abs_leg_diff = abs(ll_angle - rl_angle)
    head_y_rel = (nose[1] - y1) / bh
    com_x = np.nanmean([mid_sh[0], mid_hip[0]])
    com_x_rel = (com_x - x1) / bw

    num_valid = int(np.sum(~np.isnan(kpts[:,0])))

    return np.array([
        neck_angle, back_angle,
        lw_angle, rw_angle,
        ll_angle, rl_angle,
        beak_to_hip, body_len, left_leg_len, right_leg_len, leg_ratio,
        abs_wing_diff, abs_leg_diff,
        head_y_rel, com_x_rel,
        num_valid
    ], dtype=np.float32)

# ---------- VIDEO LOOP ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam error")
    exit()

print("✔ Running LIVE disease + advanced posture + fused health...")

SMOOTH_LEN = 7

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = yolo_chicken(frame, verbose=False)[0]

        for box in detections.boxes:
            conf = float(box.conf[0])
            if conf < 0.75:
                continue
            if int(box.cls[0]) != 0:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            if (x2-x1) < 80 or (y2-y1) < 80:
                continue

            crop = frame[y1:y2, x1:x2]

            # ---- Disease prediction ----
            try:
                c_res = cv2.resize(crop, (299, 299))
                c_norm = np.expand_dims(c_res.astype("float32")/255.0, 0)
                d_pred = disease_model.predict(c_norm, verbose=0)
                d_idx = int(np.argmax(d_pred))
                disease_name = disease_labels[d_idx]
                disease_conf = float(d_pred[0][d_idx])
            except:
                disease_name = "error"
                disease_conf = 0.0

            # ---- Posture prediction ----
            try:
                pose_res = yolo_pose(crop, verbose=False)[0]
                if len(pose_res.keypoints) > 0:
                    kpts = pose_res.keypoints.xy[0].cpu().numpy()

                    if kpts.shape[0] < 17:
                        pad = np.full((17-kpts.shape[0],2), np.nan)
                        kpts = np.vstack([kpts, pad])

                    box_pose = pose_res.boxes[0].xyxy[0].cpu().numpy().astype(int)
                    feats = compute_features_from_pose(kpts, box_pose)

                    if np.isnan(feats).any():
                        feats = np.nan_to_num(feats, nan=0.0)

                    feats_scaled = posture_scaler.transform(feats.reshape(1, -1))
                    p_pred = posture_model.predict(feats_scaled, verbose=0)
                    posture_label = posture_labels[int(np.argmax(p_pred))]
                else:
                    posture_label = "no_kp"
            except:
                posture_label = "pose_err"

            # ---- Health Fusion ----
            final_health, risk_level = fuse_health_status(disease_name, posture_label)

            # ---- Display Label ----
            text = (
                f"Disease: {disease_name} ({disease_conf:.2f}) | "
                f"Posture: {posture_label} | "
                f"Health: {final_health} {risk_level}"
            )

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1-th-12), (x1+tw+4, y1), (0,255,0), -1)
            cv2.putText(frame, text, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

        cv2.imshow("LIVE Poultry Health Monitoring", frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✔ Stopped")
