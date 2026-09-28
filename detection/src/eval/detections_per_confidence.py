import json
import argparse
from collections import Counter

def parse_args():
    p = argparse.ArgumentParser(
        description="Check how many detections per confidence band",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json", required=True, help="Path to input json with detections.")
    return p.parse_args()  # Added missing return statement


def analyze_detections(frames_data):
    empty_frames = []
    conf_bands = Counter()
    
    def get_confidence_band(conf):
        conf = max(0.0, min(1.0, conf))  # Clamp value between 0.0 and 1.0
        idx = min(int(conf * 10), 9)     # Map to bin 0-9 (1.0 goes to index 9)
        lower = idx / 10.0
        upper = (idx + 1) / 10.0
        return f"{lower:.1f} - {upper:.1f}"

    total_detections = 0

    for item in frames_data:
        frame_id = item.get("frame_id")
        detections = item.get("detections", [])
        
        if not detections:
            empty_frames.append(frame_id)
        else:
            for det in detections:
                conf = det.get("conf", 0.0)
                band = get_confidence_band(conf)
                conf_bands[band] += 1
                total_detections += 1

    print("=== DETECTION ANALYSIS ===")
    print(f"Total Frames Processed: {len(frames_data)}")
    
    print("\n=== DETECTIONS PER CONFIDENCE BAND ===")
    print(f"Total Detections: {total_detections}")
    
    # Pre-populate bins to display empty intervals cleanly
    all_bands = [f"{i/10:.1f} - {(i+1)/10:.1f}" for i in range(10)]
    for band in all_bands:
        print(f"  Band {band}: {conf_bands[band]} detection(s)")

if __name__ == "__main__":
    args = parse_args()
    with open(args.json, 'r') as f:
        data = json.load(f)
    analyze_detections(data)