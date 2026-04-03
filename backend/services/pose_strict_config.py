"""
Centralized thresholds for strict phase-AI gating and pose temporal refinement.

Citations (algorithm / API semantics — values are engineering defaults, tune with data):
- Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in
  Interactive Systems", CHI 2012 — adaptive cutoff fc = fc_min + beta * |d x/dt|,
  separate smoothing for derivative vs signal.
- Google MediaPipe Pose landmarks: visibility in [0,1]; low visibility treated as
  low confidence (see mediapipe solutions pose landmark schema).
- OpenCV VideoCapture / CAP_PROP_POS_FRAMES: seeking may not be frame-accurate on
  long-GOP codecs; this repo pairs sequential decode in keyframe_service with
  pose frame_index from the same read path (video_utils.read_frame_pose_pipeline).
"""

# --- One Euro filter (Casiez 2012) -------------------------------------------
ONE_EURO_MIN_CUTOFF_HZ = 1.0
ONE_EURO_BETA = 0.007
ONE_EURO_D_CUTOFF_HZ = 1.0

# --- Kalman constant-velocity 1D (light second stage after 1€) --------------
# Process / measurement noise: small measurement noise = trust observations more
# after 1€ already damped jitter (tracking literature: CV model for smooth motion).
KALMAN_Q = 1e-5
KALMAN_R = 4e-4

# --- Occlusion / visibility (MediaPipe visibility semantics) -----------------
VISIBILITY_LOW = 0.22
OCCLUSION_MAX_GAP_FRAMES = 4

# --- Body-scale normalization (shoulder width in normalized image coords) -----
SHOULDER_WIDTH_MIN_NORM = 0.04

# --- PoseQualityReport thresholds --------------------------------------------
COVERAGE_JOINTS = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_wrist",
    "right_wrist",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
TRUNCATION_BODY_PARTS = ("head", "left_hip", "right_hip", "left_ankle", "right_ankle")
TRUNCATION_MARGIN = 0.02

# Reliability: only "low" fails strict phase-AI (non-degraded contract).
COVERAGE_HIGH = 0.78
COVERAGE_LOW_FAIL = 0.42
TRUNCATION_SCORE_HIGH = 0.35
JITTER_SCORE_HIGH = 2.8
TRACK_CONSISTENCY_LOW = 0.55

# Jitter: median of per-frame max joint speed (normalized coords / second), wrist+shoulder
JITTER_JOINTS = ("left_wrist", "right_wrist", "left_shoulder", "right_shoulder")
TRACK_JUMP_THRESH_NORM = 0.14

# Wrist–torso jump clamp: low-visibility spikes above this (×1.5 vs track thresh) get limited.
WRIST_TORSO_JUMP_MULT = 1.5

# Post-refine vs pre-refine wrist-speed peak index delta / sequence length; above => lag warning.
SMOOTHING_LAG_HIGH_THRESH = 0.28

# Adaptive filter: low visibility → less trust in noisy observations (higher Kalman R, lower 1€ beta).
KALMAN_R_LOW_VISIBILITY_MULT = 4.0
ONE_EURO_BETA_LOW_VISIBILITY_MULT = 0.45

# Sweet spot window (frames each side of impact pose index).
# Half-width in frames: impact ± SWEET_SPOT_WINDOW (inclusive span = 2 * W + 1).
SWEET_SPOT_WINDOW = 4
SWEET_SPOT_VARIANCE_MAX_NORM = 0.035
SWEET_SPOT_CONFIDENCE_LOW = 0.35

# Gate: wrist clamps + lag imply track drift concern.
POSE_DRIFT_CLAMP_FRAMES_WARN = 6
