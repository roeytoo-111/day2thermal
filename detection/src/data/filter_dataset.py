"""
Drop a contaminated class from a YOLO dataset and remap remaining class ids
to be contiguous. Any image whose annotations become fully empty after
dropping the class is EXCLUDED ENTIRELY (image + label), not kept as a
background negative -- the evidence (area-ratio, sharpness, OCR-hit-rate,
and visual contact sheets) points to class "0" being a different image
domain entirely, not a genuine thermal background frame that just lost
its only box.

Background frames (already-empty label files) are untouched and copied over
as-is, since they're unaffected by dropping a different class.

Usage:
    python3 filter_dataset.py --dataset_dir data/thermal-1 --output_dir data/thermal-1-filtered
    python3 filter_dataset.py --dataset_dir data/thermal-1 --output_dir data/thermal-1-filtered --drop_class_name 0
"""

import os
import shutil
import glob
import argparse
import yaml

SPLITS = ["train", "valid", "test"]
IGNORED_LABEL_FILES = {"README.roboflow.txt", "classes.txt", "labels.txt"}


def parse_args():
    p = argparse.ArgumentParser(description="Drop a class from a YOLO dataset and remap the rest.")
    p.add_argument("--dataset_dir", required=True, help="Source dataset dir (with data.yaml + split subdirs).")
    p.add_argument("--output_dir", required=True, help="Where to write the filtered dataset.")
    p.add_argument("--drop_class_name", default="0", help="Class NAME to drop (default: '0').")
    return p.parse_args()


def load_names(dataset_dir):
    with open(os.path.join(dataset_dir, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    raw = cfg["names"]
    if isinstance(raw, list):
        names = {i: n for i, n in enumerate(raw)}
    else:
        names = {int(k): v for k, v in raw.items()}
    return names


def find_image(dataset_dir, split, base_name):
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(dataset_dir, split, "images", base_name + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    args = parse_args()
    names = load_names(args.dataset_dir)

    drop_id = None
    for cid, cname in names.items():
        if str(cname) == args.drop_class_name:
            drop_id = cid
            break
    if drop_id is None:
        raise ValueError(f"Class name '{args.drop_class_name}' not found in data.yaml names: {names}")

    keep_ids_sorted = sorted(cid for cid in names if cid != drop_id)
    remap = {old: new for new, old in enumerate(keep_ids_sorted)}
    new_names = [names[cid] for cid in keep_ids_sorted]

    print(f"Dropping class id {drop_id} ('{args.drop_class_name}').")
    print(f"Remap (old id -> new id): {remap}")
    print(f"New class list: {new_names}\n")

    stats = {
        "kept_images_with_boxes": 0,
        "kept_background_images": 0,
        "dropped_images_all_target_class": 0,
        "dropped_boxes_of_target_class": 0,
        "images_missing_on_disk": 0,
    }

    for split in SPLITS:
        in_labels_dir = os.path.join(args.dataset_dir, split, "labels")
        out_labels_dir = os.path.join(args.output_dir, split, "labels")
        out_images_dir = os.path.join(args.output_dir, split, "images")
        os.makedirs(out_labels_dir, exist_ok=True)
        os.makedirs(out_images_dir, exist_ok=True)

        for label_path in glob.glob(os.path.join(in_labels_dir, "*.txt")):
            fname = os.path.basename(label_path)
            if fname in IGNORED_LABEL_FILES or fname.startswith("README"):
                continue
            base = os.path.splitext(fname)[0]
            img_path = find_image(args.dataset_dir, split, base)

            if os.path.getsize(label_path) == 0:
                # Genuine background frame, unaffected by dropping a different class.
                if img_path:
                    shutil.copy(img_path, os.path.join(out_images_dir, os.path.basename(img_path)))
                    open(os.path.join(out_labels_dir, fname), "w").close()
                    stats["kept_background_images"] += 1
                else:
                    stats["images_missing_on_disk"] += 1
                continue

            kept_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5 or not parts[0].isdigit():
                        continue
                    cid = int(parts[0])
                    if cid == drop_id:
                        stats["dropped_boxes_of_target_class"] += 1
                        continue
                    new_cid = remap[cid]
                    kept_lines.append(" ".join([str(new_cid)] + parts[1:]))

            if not kept_lines:
                # Every box in this image was the dropped class -> exclude the
                # image entirely, don't relabel it as background.
                stats["dropped_images_all_target_class"] += 1
                continue

            if img_path:
                shutil.copy(img_path, os.path.join(out_images_dir, os.path.basename(img_path)))
            else:
                stats["images_missing_on_disk"] += 1
                continue
            with open(os.path.join(out_labels_dir, fname), "w") as f:
                f.write("\n".join(kept_lines) + "\n")
            stats["kept_images_with_boxes"] += 1

    new_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(new_names),
        "names": new_names,
    }
    with open(os.path.join(args.output_dir, "data.yaml"), "w") as f:
        yaml.safe_dump(new_yaml, f, sort_keys=False)

    print("Done. Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nFiltered dataset written to: {args.output_dir}")


if __name__ == "__main__":
    main()