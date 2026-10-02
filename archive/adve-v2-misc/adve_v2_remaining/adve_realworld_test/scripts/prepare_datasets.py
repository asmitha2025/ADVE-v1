import os
import glob
import json
import argparse


def generate_manifest(video_dir: str, output_manifest: str):
    print(f"=== Generating Dataset Manifest from '{video_dir}' ===")
    
    video_extensions = ["*.mp4", "*.webm", "*.avi", "*.mkv"]
    video_files = []
    if os.path.isfile(video_dir):
        video_files.append(video_dir)
    elif os.path.isdir(video_dir):
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(video_dir, "**", ext), recursive=True))

    dataset_entries = []
    for v_path in video_files:
        abs_path = os.path.abspath(v_path)
        base_name = os.path.basename(v_path)
        name_no_ext = os.path.splitext(base_name)[0]

        # Domain heuristic detection
        domain = "general"
        if "mot" in name_no_ext.lower() or "street" in name_no_ext.lower():
            domain = "urban_surveillance"
        elif "office" in name_no_ext.lower() or "store" in name_no_ext.lower() or "retail" in name_no_ext.lower():
            domain = "retail_indoor"
        elif "night" in name_no_ext.lower() or "dark" in name_no_ext.lower():
            domain = "low_light"
        elif "highway" in name_no_ext.lower() or "traffic" in name_no_ext.lower():
            domain = "traffic_monitoring"

        dataset_entries.append({
            "name": name_no_ext,
            "path": abs_path,
            "domain": domain,
            "lighting": "daylight" if domain != "low_light" else "low_light"
        })

    manifest = {"datasets": dataset_entries}
    os.makedirs(os.path.dirname(os.path.abspath(output_manifest)), exist_ok=True)
    with open(output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Created manifest with {len(dataset_entries)} videos -> '{output_manifest}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-manifest", required=True, help="Directory containing videos")
    parser.add_argument("--output", default="datasets/my_manifest.json")
    args = parser.parse_args()

    generate_manifest(args.generate_manifest, args.output)
