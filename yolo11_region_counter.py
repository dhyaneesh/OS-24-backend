import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

counting_regions = [
    {
        "name": "Polygon Region",  # First custom-shaped region
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),
        "counts": 0,  # Counter for people in this region
        "log": []  # Log for entry and exit events
    },
    {
        "name": "Rectangle Region",  # Second rectangular-like region
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
        "counts": 0,  # Counter for people in this region
        "log": []  # Log for entry and exit events
    },
]

REGION_COLORS = [
    tuple(np.random.randint(0, 255, 3).tolist()) for _ in counting_regions
]

def run(
    weights="yolo11n.pt",  # Path to YOLO model weights
    source=None,  # Path to the input video
    device="gpu",  # Specify processing device (CPU/GPU)
    view_img=False,  # Flag to display the processed video frames
    save_img=False,  # Flag to save the processed video
    classes=[0],  # Class index to detect (0 corresponds to "person" in COCO dataset)
    line_thickness=2,  # Thickness of bounding boxes
    region_thickness=2,  # Thickness of region boundaries
):
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    model = YOLO(f"{weights}")
    names = model.model.names
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "Processed_Output.mp4"
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    object_paths = {}

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break

        results = model.track(frame, persist=True, classes=classes)
        frame_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, cls, obj_id in zip(boxes, clss, ids):
                if cls == 0:
                    annotator.box_label(box, str(int(obj_id)), color=colors(cls, True))
                    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                    if obj_id not in object_paths:
                        object_paths[obj_id] = []
                    object_paths[obj_id].append(bbox_center)

                    for i, region in enumerate(counting_regions):
                        # REGION Definition Logic
                        region_color = REGION_COLORS[i % len(REGION_COLORS)]
                        polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
                        cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

                        # Person count logic 
                        person_count = sum(
                            region["polygon"].contains(bbox_center)
                            for box in boxes
                        )
                        region["counts"] = person_count
                        region_center = tuple(np.mean(polygon_coords, axis=0).astype(int))
                        cv2.putText(
                            frame, f"Count: {person_count}", region_center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, region_color, 2
                        )
                        
                        # ENTRY/EXIT Logic
                        if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                            if not any(log_entry["id"] == obj_id and log_entry["action"] == "ENTRY" for log_entry in region["log"]):
                                region["log"].append({"id": obj_id, "action": "ENTRY", "time": frame_time})
                        else:
                            if any(log_entry["id"] == obj_id and log_entry["action"] == "ENTRY" for log_entry in region["log"]):
                                region["log"].append({"id": obj_id, "action": "EXIT", "time": frame_time})

        print("Counting Regions:")
        for region in counting_regions:
            print(f'REGION: {region["name"]}: {region["counts"]}')
        
        total_count = len(ids)
        cv2.putText(
            frame, f"Total Count: {total_count}", (frame_width // 2 - 100, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        if view_img:
            cv2.imshow("Region Counter", frame)

        if save_img:
            video_writer.write(frame)

        for region in counting_regions:
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()

def main():
    run(
        weights="yolo11n.pt",
        source="../data/output_4.mp4",
        device="gpu",
        view_img=True,
        save_img=True,
        line_thickness=2,
        region_thickness=2,
    )

if __name__ == "__main__":
    main()
