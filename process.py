import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
from db import create_region, update_log_event, get_regions, get_logs, get_active_log, get_region_count

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

counting_regions = [
    {
        "name": "Region 1",
        "x_min": 50,
        "y_min": 80,
        "x_max": 450,
        "y_max": 350,
        "count": 0
    },
    {
        "name": "Region 2",
        "x_min": 200,
        "y_min": 250,
        "x_max": 440,
        "y_max": 550,
        "count": 0
    },
]

regions = []

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
    
    for region in counting_regions:
        created_region = create_region(region["name"], region["x_min"], region["y_min"], region["x_max"], region["y_max"])
    
    regions = get_regions()

    print(regions)

    model = YOLO(f"{weights}")

    model.to("mps")

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
        frame_time = datetime.now()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, cls, obj_id in zip(boxes, clss, ids):
                if cls == 0:
                    annotator.box_label(box, str(int(obj_id)), color=colors(cls, True))
                    bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

                    if obj_id not in object_paths:
                        object_paths[obj_id] = []
                    object_paths[obj_id].append(bbox_center)

                    for i, region in enumerate(regions):
                        region_color = REGION_COLORS[i % len(REGION_COLORS)]
                        cv2.rectangle(
                            frame,
                            (region["x_min"], region["y_min"]),
                            (region["x_max"], region["y_max"]),
                            region_color,
                            thickness=region_thickness
                        )

                        point = Point(bbox_center[0], bbox_center[1])  # Create a point for the object's center
                        polygon = Polygon([
                            (region["x_min"], region["y_min"]),
                            (region["x_max"], region["y_min"]),
                            (region["x_max"], region["y_max"]),
                            (region["x_min"], region["y_max"])
                        ])  # Create a polygon for the region

                        log = get_active_log(region['name'], obj_id)
                        
                        # ENTRY/EXIT Logic
                        if point.within(polygon):
                            if not log:
                                update_log_event(region["name"], obj_id)
                        else:
                            if log:
                                update_log_event(region["name"], obj_id)

        print("Counting Regions:")
        for region in regions:
            print(f'REGION: {region["name"]}: {get_region_count(region["name"])}')
        
        total_count = len(ids)
        cv2.putText(
            frame, f"Total Count: {total_count}", (frame_width // 2 - 100, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        if view_img:
            cv2.imshow("Region Counter", frame)

        if save_img:
            video_writer.write(frame)

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