import copy
import numpy as np
from utils.utils_calib import pan_tilt_roll_to_orientation


class TemporalStabilizer:
    """
    Temporal smoother for PnLCalib camera parameters.
    Works like an EMA with outlier rejection.
    """
    def __init__(
        self,
        alpha=0.35,               # EMA "learning rate" (0 = frozen, 1 = no smoothing)
        alpha_high_err=0.08,      # more conservative when reprojection error is high
        max_pos_jump=20.0,        # meters; broadcast cams don't teleport 20m in 1 frame
        max_angle_jump=15.0,      # degrees
        max_focal_jump=800.0,     # pixels
        max_rep_err=8.0,          # px threshold for "good" frame
        reject_rep_err=15.0,      # px threshold for hard rejection
        reset_after_reject=5,     # allow reset after N consecutive rejects (scene cut)
    ):
        self.alpha = alpha
        self.alpha_high_err = alpha_high_err
        self.max_pos_jump = max_pos_jump
        self.max_angle_jump = max_angle_jump
        self.max_focal_jump = max_focal_jump
        self.max_rep_err = max_rep_err
        self.reject_rep_err = reject_rep_err
        self.reset_after_reject = reset_after_reject

        self.last_valid = None
        self.reject_count = 0

    def smooth(self, result_dict):
        """
        result_dict: output of FramebyFrameCalib.heuristic_voting()
        Returns: stabilized result_dict (or previous one if current is rejected)
        """
        if result_dict is None:
            return self.last_valid

        cam = result_dict["cam_params"]
        rep_err = result_dict.get("rep_err", 999.0)

        # First frame or first valid frame
        if self.last_valid is None:
            self.last_valid = copy.deepcopy(result_dict)
            self.reject_count = 0
            return result_dict

        prev_cam = self.last_valid["cam_params"]

        # ---- 1. Outlier check ----
        valid, reason = self._is_valid(cam, prev_cam, rep_err)

        if not valid and rep_err > self.reject_rep_err:
            self.reject_count += 1
            if self.reject_count >= self.reset_after_reject:
                # Likely a scene cut; accept and reset history
                self.last_valid = copy.deepcopy(result_dict)
                self.reject_count = 0
                return result_dict
            # Reject: reuse last valid frame
            return self.last_valid

        # ---- 2. Choose blend factor ----
        if rep_err > self.max_rep_err or not valid:
            alpha = self.alpha_high_err
        else:
            alpha = self.alpha

        # ---- 3. EMA blend ----
        smoothed_cam = self._blend(prev_cam, cam, alpha)

        smoothed_dict = copy.deepcopy(result_dict)
        smoothed_dict["cam_params"] = smoothed_cam
        # Optional: flag that this was smoothed
        # smoothed_dict["smoothed"] = True

        self.last_valid = smoothed_dict
        self.reject_count = 0
        return smoothed_dict

    def _is_valid(self, curr, prev, rep_err):
        # Position jump
        pos_dist = np.linalg.norm(
            np.array(curr["position_meters"]) - np.array(prev["position_meters"])
        )
        if pos_dist > self.max_pos_jump:
            return False, f"pos_jump={pos_dist:.1f}m"

        # Angle jumps (handle pan wrap-around)
        pan_diff = abs(((curr["pan_degrees"] - prev["pan_degrees"] + 180) % 360) - 180)
        tilt_diff = abs(curr["tilt_degrees"] - prev["tilt_degrees"])
        roll_diff = abs(curr["roll_degrees"] - prev["roll_degrees"])
        if (
            pan_diff > self.max_angle_jump
            or tilt_diff > self.max_angle_jump
            or roll_diff > self.max_angle_jump
        ):
            return False, f"angle pan={pan_diff:.1f} tilt={tilt_diff:.1f} roll={roll_diff:.1f}"

        # Focal length jump
        focal_diff = abs(curr["x_focal_length"] - prev["x_focal_length"])
        if focal_diff > self.max_focal_jump:
            return False, f"focal_jump={focal_diff:.1f}px"

        return True, "ok"

    def _blend(self, prev, curr, alpha):
        out = copy.deepcopy(curr)

        # Position
        for i in range(3):
            out["position_meters"][i] = (
                prev["position_meters"][i]
                + alpha * (curr["position_meters"][i] - prev["position_meters"][i])
            )

        # Angles
        out["pan_degrees"] = self._angle_blend(
            prev["pan_degrees"], curr["pan_degrees"], alpha
        )
        out["tilt_degrees"] = (
            prev["tilt_degrees"] + alpha * (curr["tilt_degrees"] - prev["tilt_degrees"])
        )
        out["roll_degrees"] = (
            prev["roll_degrees"] + alpha * (curr["roll_degrees"] - prev["roll_degrees"])
        )

        # Intrinsics
        out["x_focal_length"] = (
            prev["x_focal_length"]
            + alpha * (curr["x_focal_length"] - prev["x_focal_length"])
        )
        out["y_focal_length"] = (
            prev["y_focal_length"]
            + alpha * (curr["y_focal_length"] - prev["y_focal_length"])
        )
        out["principal_point"][0] = (
            prev["principal_point"][0]
            + alpha * (curr["principal_point"][0] - prev["principal_point"][0])
        )
        out["principal_point"][1] = (
            prev["principal_point"][1]
            + alpha * (curr["principal_point"][1] - prev["principal_point"][1])
        )

        # Rebuild rotation matrix from smoothed Euler angles
        R = pan_tilt_roll_to_orientation(
            np.deg2rad(out["pan_degrees"]),
            np.deg2rad(out["tilt_degrees"]),
            np.deg2rad(out["roll_degrees"]),
        )
        out["rotation_matrix"] = R.tolist()
        return out

    @staticmethod
    def _angle_blend(a1, a2, alpha):
        """Blend angles with 360° wrap-around handling."""
        diff = ((a2 - a1 + 180) % 360) - 180
        return (a1 + alpha * diff + 360) % 360
