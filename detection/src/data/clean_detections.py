"""
Clean the RGB detection JSON into a plausible single-target track by gating
out frame-to-frame jumps that are too large to be real motion (or too large a
box-size change), using thresholds derived from the data's own empirical
per-frame motion -- not a hand-picked pixel constant that may not generalize
across zoom levels/altitudes.

Only 'confirmed' and 'accepted_high_conf' detections are considered as track
candidates ('rejected' is already excluded, since the two-stage pipeline
marked those as likely false positives).

Approach: a simple forward gating pass (à la a minimal single-target SORT).
State = last accepted center + a smoothed velocity estimate. For each new
candidate: predict where the target should be, and accept only if the
candidate falls within an adaptive gate that grows with the time gap since
the last accepted point. A long run of consecutive rejects forces a track
reset (re-acquisition) rather than getting stuck matching against a stale
prediction forever.

Usage:
    python3 clean_detections.py --json_path detection_tiled_....json --output_dir gt_track_out
"""

import json
import math
import argparse
import statistics

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Gate implausible jumps out of the RGB detection track.")
    p.add_argument("--json_path", required=True)
    p.add_argument("--output_dir", default="./gt_track_out")
    p.add_argument("--gate_k", type=float, default=8.0,
                    help="Per-frame speed gate = median + gate_k * MAD of observed gap==1 speeds (default: 8.0). "
                         "Lower = stricter (rejects more), higher = more permissive.")
    p.add_argument("--gate_floor_px", type=float, default=5.0,
                    help="Minimum per-frame speed gate in px, in case MAD is ~0 (default: 5.0).")
    p.add_argument("--max_track_gap_frames", type=int, default=30,
                    help="If more than this many frames pass with no accepted point, force a track reset "
                         "instead of predicting across the gap (default: 30, i.e. ~1s at 30fps).")
    p.add_argument("--max_consecutive_outliers", type=int, default=10,
                    help="Force a track reset after this many rejected candidates in a row (default: 10) "
                         "-- protects against a stale/wrong prediction rejecting everything forever.")
    p.add_argument("--area_log2_ratio_gate", type=float, default=3.0,
                    help="Reject if |log2(area_next/area_prev)| exceeds this even when position passes "
                         "(default: 3.0, i.e. an 8x box-size jump in one accepted step).")
    p.add_argument("--velocity_alpha", type=float, default=0.5,
                    help="Exponential smoothing factor for the velocity estimate (default: 0.5).")
    return p.parse_args()


def load_candidates(json_path):
    with open(json_path) as f:
        data = json.load(f)
    rows = []
    for entry in data:
        for det in entry["detections"]:
            if det["final"] == "rejected":
                continue
            x1, y1, x2, y2 = det["bbox"]
            rows.append({
                "frame_id": entry["frame_id"],
                "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                "w": x2 - x1, "h": y2 - y1,
                "area": max(1.0, (x2 - x1) * (y2 - y1)),
                "stage1_conf": det["stage1_conf"], "stage2_prob": det["stage2_prob"],
                "final": det["final"],
                "bbox": det["bbox"],
            })
    rows.sort(key=lambda r: r["frame_id"])
    return rows


def estimate_speed_gate(candidates, gate_k, gate_floor_px):
    """Derive the per-frame pixel-speed gate from the data's own gap==1 transitions."""
    speeds = []
    for a, b in zip(candidates, candidates[1:]):
        gap = b["frame_id"] - a["frame_id"]
        if gap != 1:
            continue
        d = math.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"])
        speeds.append(d)
    if not speeds:
        return gate_floor_px
    med = statistics.median(speeds)
    mad = statistics.median([abs(s - med) for s in speeds]) or 1.0
    gate = med + gate_k * mad
    return max(gate, gate_floor_px), med, mad


def clean_track(candidates, args):
    gate_per_frame, med_speed, mad_speed = estimate_speed_gate(candidates, args.gate_k, args.gate_floor_px)
    print(f"Empirical gap==1 speed: median={med_speed:.2f}px/frame, MAD={mad_speed:.2f}px/frame")
    print(f"Derived per-frame speed gate: {gate_per_frame:.2f}px/frame (gate_k={args.gate_k})")

    accepted = []
    rejected = []
    track_events = []  # log of resets

    if not candidates:
        return accepted, rejected, track_events

    # seed track with the first candidate
    state = {
        "cx": candidates[0]["cx"], "cy": candidates[0]["cy"],
        "area": candidates[0]["area"], "vx": 0.0, "vy": 0.0,
        "last_frame": candidates[0]["frame_id"],
    }
    accepted.append({**candidates[0], "track_status": "seed"})
    consecutive_outliers = 0

    for cand in candidates[1:]:
        gap = cand["frame_id"] - state["last_frame"]
        if gap <= 0:
            continue  # duplicate/out-of-order frame_id, skip defensively

        if gap > args.max_track_gap_frames:
            # too long since last accepted point to trust a prediction -- reset
            accepted.append({**cand, "track_status": "reacquired_after_gap"})
            track_events.append({"frame_id": cand["frame_id"], "event": "reset_long_gap", "gap": gap})
            state = {"cx": cand["cx"], "cy": cand["cy"], "area": cand["area"],
                     "vx": 0.0, "vy": 0.0, "last_frame": cand["frame_id"]}
            consecutive_outliers = 0
            continue

        pred_cx = state["cx"] + state["vx"] * gap
        pred_cy = state["cy"] + state["vy"] * gap
        dist = math.hypot(cand["cx"] - pred_cx, cand["cy"] - pred_cy)
        allowed = gate_per_frame * gap

        area_log2_ratio = math.log2(cand["area"] / state["area"]) if state["area"] > 0 else 0.0

        if dist <= allowed and abs(area_log2_ratio) <= args.area_log2_ratio_gate:
            new_vx = (cand["cx"] - state["cx"]) / gap
            new_vy = (cand["cy"] - state["cy"]) / gap
            a = args.velocity_alpha
            state["vx"] = a * new_vx + (1 - a) * state["vx"]
            state["vy"] = a * new_vy + (1 - a) * state["vy"]
            state["cx"], state["cy"], state["area"] = cand["cx"], cand["cy"], cand["area"]
            state["last_frame"] = cand["frame_id"]
            accepted.append({**cand, "track_status": "accepted"})
            consecutive_outliers = 0
        else:
            reason = "position_jump" if dist > allowed else "size_jump"
            rejected.append({**cand, "reject_reason": reason, "dist_px": round(dist, 1),
                              "allowed_px": round(allowed, 1), "area_log2_ratio": round(area_log2_ratio, 2)})
            consecutive_outliers += 1
            if consecutive_outliers >= args.max_consecutive_outliers:
                # prediction is probably stale -- reacquire on this point rather
                # than keep rejecting everything against a dead trajectory
                accepted.append({**cand, "track_status": "reacquired_after_outliers"})
                track_events.append({"frame_id": cand["frame_id"], "event": "reset_consecutive_outliers"})
                state = {"cx": cand["cx"], "cy": cand["cy"], "area": cand["area"],
                         "vx": 0.0, "vy": 0.0, "last_frame": cand["frame_id"]}
                consecutive_outliers = 0

    return accepted, rejected, track_events


def main():
    args = parse_args()
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    candidates = load_candidates(args.json_path)
    print(f"Loaded {len(candidates)} candidate detections (confirmed + accepted_high_conf) "
          f"across frames {candidates[0]['frame_id']}-{candidates[-1]['frame_id']}.")

    accepted, rejected, events = clean_track(candidates, args)

    acc_df = pd.DataFrame(accepted)
    rej_df = pd.DataFrame(rejected)
    ev_df = pd.DataFrame(events)

    acc_csv = f"{args.output_dir}/cleaned_track.csv"
    rej_csv = f"{args.output_dir}/rejected_jumps.csv"
    acc_df.to_csv(acc_csv, index=False)
    rej_df.to_csv(rej_csv, index=False)
    if not ev_df.empty:
        ev_df.to_csv(f"{args.output_dir}/track_reset_events.csv", index=False)

    print("\n" + "=" * 74)
    print("                          RESULTS")
    print("=" * 74)
    print(f"Accepted (track-consistent): {len(acc_df)}")
    print(f"Rejected (implausible jump/size change): {len(rej_df)}")
    if not rej_df.empty:
        print("\nRejection reason breakdown:")
        print(rej_df["reject_reason"].value_counts().to_string())
    print(f"\nTrack resets: {len(ev_df)}")
    print(f"\nAccepted track spans frame {acc_df['frame_id'].min()} to {acc_df['frame_id'].max()} "
          f"-- this brackets the actual flight window within the RGB video.")
    print(f"\nSaved: {acc_csv}")
    print(f"Saved: {rej_csv}")


if __name__ == "__main__":
    main()