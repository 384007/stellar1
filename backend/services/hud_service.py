from typing import Optional

from services.handedness_service import (
    enrich_angles_with_lead_trail,
    get_lead_trail_sides,
    normalize_hand,
)


def generate_hud_data(
    pose_data: dict, mode: str = "lite", hand: str = "UNKNOWN"
) -> dict:
    """
    Transform raw pose data into HUD-ready display data.
    mode: 'lite' shows 4 default joints, 'pro' shows all 8.
    """
    DEFAULT_VISIBLE = {"left_hip", "right_hip", "left_knee", "right_shoulder", "left_shoulder"}
    # Pro: full stick figure incl. head, both arms, both legs (upper / lower calibration)
    EXTENDED_VISIBLE = {
        "head",
        "left_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
    }

    if mode == "pro":
        visible_joints = DEFAULT_VISIBLE | EXTENDED_VISIBLE
    else:
        visible_joints = DEFAULT_VISIBLE

    joints = pose_data.get("joints", [])
    resolved_hand = normalize_hand(hand)
    angles = enrich_angles_with_lead_trail(pose_data.get("angles", {}), resolved_hand)
    side_map = get_lead_trail_sides(resolved_hand)
    lead_side = side_map["lead_side"]
    trail_side = side_map["trail_side"]
    connections = pose_data.get("connections", [])
    frame_size = pose_data.get("frame_size", {"width": 640, "height": 480})

    hud_joints = []
    for joint in joints:
        is_default = joint["name"] in DEFAULT_VISIBLE
        is_extended = joint["name"] in EXTENDED_VISIBLE
        is_visible = joint["name"] in visible_joints

        angle_value = angles.get(joint["name"])
        upper = {
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        }
        region = (
            "head"
            if joint["name"] == "head"
            else ("upper" if joint["name"] in upper else "lower")
        )

        hud_joints.append(
            {
                "name": joint["name"],
                "x": joint.get("normalized", {}).get("x", joint["x"] / frame_size["width"]),
                "y": joint.get("normalized", {}).get("y", joint["y"] / frame_size["height"]),
                "visible": is_visible,
                "region": region,
                "category": "default" if is_default else ("extended" if is_extended else "hidden"),
                "angle": angle_value,
                "semantic_role": _joint_semantic_role(joint["name"], lead_side, trail_side),
                "color": _get_joint_color(joint["name"], angle_value),
                "pulse": is_visible,
            }
        )

    hud_connections = []
    joint_names = [j["name"] for j in joints]
    for conn in connections:
        if conn[0] < len(joint_names) and conn[1] < len(joint_names):
            from_name = joint_names[conn[0]]
            to_name = joint_names[conn[1]]
            from_visible = from_name in visible_joints
            to_visible = to_name in visible_joints
            hud_connections.append(
                {
                    "from": conn[0],
                    "to": conn[1],
                    "from_name": from_name,
                    "to_name": to_name,
                    "visible": from_visible and to_visible,
                    "gradient": _get_connection_gradient(from_name, to_name),
                }
            )

    return {
        "joints": hud_joints,
        "connections": hud_connections,
        "angles": angles,
        "frame_size": frame_size,
        "mode": mode,
        "stats": {
            "shoulder_rotation": angles.get("shoulder_rotation", 0),
            "hip_rotation": angles.get("hip_rotation", 0),
            "x_factor": angles.get("x_factor", 0),
            "spine_tilt": angles.get("spine_tilt", 0),
            "hand": resolved_hand,
            "lead_side": lead_side,
            "trail_side": trail_side,
            "lead_elbow": angles.get("lead_elbow", 0),
            "trail_elbow": angles.get("trail_elbow", 0),
            "lead_knee": angles.get("lead_knee", 0),
            "trail_knee": angles.get("trail_knee", 0),
        },
    }


def _get_joint_color(name: str, angle: Optional[float]) -> str:
    if angle is None:
        return "#00ff88"

    upper_body = {"head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}
    if name in upper_body:
        return "#00e5ff"
    return "#00ff88"


def _get_connection_gradient(from_name: str, to_name: str) -> list[str]:
    upper = {"head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}
    if from_name in upper and to_name in upper:
        return ["#00e5ff", "#00bcd4"]
    elif from_name in upper or to_name in upper:
        return ["#00e5ff", "#00ff88"]
    return ["#00ff88", "#00c853"]


def _joint_semantic_role(name: str, lead_side: str, trail_side: str) -> str:
    if name == f"{lead_side}_wrist":
        return "lead_arm"
    if name == f"{trail_side}_wrist":
        return "trail_arm"
    if name == f"{lead_side}_knee":
        return "lead_leg"
    if name == f"{trail_side}_knee":
        return "trail_leg"
    if name == f"{lead_side}_shoulder":
        return "lead_shoulder"
    if name == f"{trail_side}_shoulder":
        return "trail_shoulder"
    if name == f"{lead_side}_hip":
        return "lead_hip"
    if name == f"{trail_side}_hip":
        return "trail_hip"
    return "neutral"
