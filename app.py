# app.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    SOURCE_PATH,
    OUTPUT_PATH,
    CONFIDENCE_THRESHOLD,
    NMS_IOU_THRESHOLD,
    IMAGE_SIZE,
    DEVICE,
    TRACKER_CONFIG,
    CLASS_NAMES,
    LINE_LEFT_START,
    LINE_LEFT_END,
    LEFT_REGION_X_MIN,
    LEFT_REGION_X_MAX,
    LINE_RIGHT_START,
    LINE_RIGHT_END,
    RIGHT_REGION_X_MIN,
    RIGHT_REGION_X_MAX,
    MIN_BOX_AREA,
    MAX_BOX_AREA_RATIO,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    DUPLICATE_IOU_THRESHOLD,
    DUPLICATE_CENTER_DISTANCE,
    COUNT_EVENT_COOLDOWN_FRAMES,
    COUNT_EVENT_COOLDOWN_DISTANCE,
    SHOW_WINDOW,
    WINDOW_NAME,
    LINE_LEFT_COLOR,
    LINE_RIGHT_COLOR,
    BOX_COLOR,
    CENTER_COLOR,
    TEXT_COLOR,
    LEFT_TEXT_COLOR,
    RIGHT_TEXT_COLOR,
    LINE_THICKNESS,
    FONT_SCALE,
    FONT_THICKNESS,
)


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    center: tuple[int, int]
    class_id: int
    class_name: str
    track_id: int
    confidence: float


@dataclass(frozen=True)
class CountEvent:
    frame_index: int
    line_name: str
    class_name: str
    center: tuple[int, int]


def ensure_output_directory(output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def validate_counting_lines() -> None:
    if LINE_LEFT_START[1] != LINE_LEFT_END[1]:
        raise ValueError(
            "A linha da esquerda deve ser horizontal. "
            "Defina LINE_LEFT_START e LINE_LEFT_END com o mesmo valor de Y."
        )

    if LINE_RIGHT_START[1] != LINE_RIGHT_END[1]:
        raise ValueError(
            "A linha da direita deve ser horizontal. "
            "Defina LINE_RIGHT_START e LINE_RIGHT_END com o mesmo valor de Y."
        )


def open_video_capture(source_path: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source_path)

    if not capture.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir o vídeo de entrada: {source_path}")

    return capture


def create_video_writer(
    output_path: str,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps,
                             (frame_width, frame_height))

    if not writer.isOpened():
        raise RuntimeError(
            f"Não foi possível criar o vídeo de saída: {output_path}")

    return writer


def get_video_properties(capture: cv2.VideoCapture) -> tuple[float, int, int]:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return fps, frame_width, frame_height


def get_box_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def calculate_box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = calculate_box_area(box_a)
    area_b = calculate_box_area(box_b)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def calculate_center_distance(
    center_a: tuple[int, int],
    center_b: tuple[int, int],
) -> float:
    return hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])


def is_valid_detection(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> bool:
    area = calculate_box_area(box)

    if area < MIN_BOX_AREA:
        return False

    if area > frame_width * frame_height * MAX_BOX_AREA_RATIO:
        return False

    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect_ratio = width / height

    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False

    return True


def remove_same_frame_duplicates(detections: list[Detection]) -> list[Detection]:
    if not detections:
        return []

    detections_sorted = sorted(
        detections,
        key=lambda detection: detection.confidence,
        reverse=True,
    )

    filtered_detections: list[Detection] = []

    for detection in detections_sorted:
        should_keep = True

        for kept_detection in filtered_detections:
            if detection.class_id != kept_detection.class_id:
                continue

            iou = calculate_iou(detection.box, kept_detection.box)
            distance = calculate_center_distance(
                detection.center,
                kept_detection.center,
            )

            if iou >= DUPLICATE_IOU_THRESHOLD or distance <= DUPLICATE_CENTER_DISTANCE:
                should_keep = False
                break

        if should_keep:
            filtered_detections.append(detection)

    return filtered_detections


def did_cross_horizontal_line(
    previous_center: tuple[int, int],
    current_center: tuple[int, int],
    line_y: int,
) -> bool:
    previous_y = previous_center[1]
    current_y = current_center[1]

    crossed_downward = previous_y < line_y <= current_y
    crossed_upward = previous_y > line_y >= current_y

    return crossed_downward or crossed_upward


def is_inside_x_region(
    center_x: int,
    x_min: int,
    x_max: int,
) -> bool:
    return x_min <= center_x <= x_max


def is_duplicate_count_event(
    recent_events: deque[CountEvent],
    current_frame_index: int,
    line_name: str,
    class_name: str,
    center: tuple[int, int],
) -> bool:
    for event in recent_events:
        is_same_line = event.line_name == line_name
        is_same_class = event.class_name == class_name
        is_within_cooldown = (
            current_frame_index - event.frame_index <= COUNT_EVENT_COOLDOWN_FRAMES
        )
        is_nearby = (
            calculate_center_distance(center, event.center)
            <= COUNT_EVENT_COOLDOWN_DISTANCE
        )

        if is_same_line and is_same_class and is_within_cooldown and is_nearby:
            return True

    return False


def extract_detections(
    result,
    frame_width: int,
    frame_height: int,
) -> list[Detection]:
    if result.boxes is None or result.boxes.id is None:
        return []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
    boxes_id = result.boxes.id.cpu().numpy().astype(int)
    boxes_conf = result.boxes.conf.cpu().numpy()

    detections: list[Detection] = []

    for box, class_id, track_id, confidence in zip(
        boxes_xyxy,
        boxes_cls,
        boxes_id,
        boxes_conf,
    ):
        if class_id not in CLASS_NAMES:
            continue

        x1, y1, x2, y2 = map(int, box)
        current_box = (x1, y1, x2, y2)

        if not is_valid_detection(current_box, frame_width, frame_height):
            continue

        detections.append(
            Detection(
                box=current_box,
                center=get_box_center(x1, y1, x2, y2),
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                track_id=int(track_id),
                confidence=float(confidence),
            )
        )

    return remove_same_frame_duplicates(detections)


def draw_counting_lines(frame) -> None:
    cv2.line(
        frame,
        LINE_LEFT_START,
        LINE_LEFT_END,
        LINE_LEFT_COLOR,
        LINE_THICKNESS,
    )
    cv2.line(
        frame,
        LINE_RIGHT_START,
        LINE_RIGHT_END,
        LINE_RIGHT_COLOR,
        LINE_THICKNESS,
    )

    cv2.putText(
        frame,
        "LEFT FLOW",
        (LINE_LEFT_START[0], max(25, LINE_LEFT_START[1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        LINE_LEFT_COLOR,
        2,
    )
    cv2.putText(
        frame,
        "RIGHT FLOW",
        (LINE_RIGHT_START[0], max(25, LINE_RIGHT_START[1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        LINE_RIGHT_COLOR,
        2,
    )


def draw_detection(frame, detection: Detection) -> None:
    x1, y1, x2, y2 = detection.box

    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    cv2.circle(frame, detection.center, 4, CENTER_COLOR, -1)

    label = f"{detection.class_name} ID:{detection.track_id} {detection.confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
    )


def draw_counts(
    frame,
    left_counts: dict[str, int],
    right_counts: dict[str, int],
) -> None:
    y_position = 30

    cv2.putText(
        frame,
        "Contagem LEFT FLOW",
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        LEFT_TEXT_COLOR,
        2,
    )
    y_position += 30

    for class_name, total in left_counts.items():
        cv2.putText(
            frame,
            f"{class_name}: {total}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            LEFT_TEXT_COLOR,
            2,
        )
        y_position += 28

    y_position += 15

    cv2.putText(
        frame,
        "Contagem RIGHT FLOW",
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        RIGHT_TEXT_COLOR,
        2,
    )
    y_position += 30

    for class_name, total in right_counts.items():
        cv2.putText(
            frame,
            f"{class_name}: {total}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RIGHT_TEXT_COLOR,
            2,
        )
        y_position += 28


def print_final_counts(title: str, counts: dict[str, int]) -> None:
    print(f"\n{title}")
    for class_name, total in counts.items():
        print(f"{class_name}: {total}")


def run() -> None:
    validate_counting_lines()
    ensure_output_directory(OUTPUT_PATH)

    model = YOLO(MODEL_PATH)
    capture = open_video_capture(SOURCE_PATH)
    fps, frame_width, frame_height = get_video_properties(capture)
    writer = create_video_writer(OUTPUT_PATH, fps, frame_width, frame_height)

    allowed_classes = list(CLASS_NAMES.keys())

    left_line_y = LINE_LEFT_START[1]
    right_line_y = LINE_RIGHT_START[1]

    last_centers_by_track_id: dict[int, tuple[int, int]] = {}

    counted_on_left_line_ids: set[int] = set()
    counted_on_right_line_ids: set[int] = set()

    left_counts = {class_name: 0 for class_name in CLASS_NAMES.values()}
    right_counts = {class_name: 0 for class_name in CLASS_NAMES.values()}

    recent_count_events: deque[CountEvent] = deque(maxlen=300)
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1

            results = model.track(
                source=frame,
                persist=True,
                conf=CONFIDENCE_THRESHOLD,
                iou=NMS_IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                tracker=TRACKER_CONFIG,
                classes=allowed_classes,
                verbose=False,
            )

            annotated_frame = frame.copy()
            draw_counting_lines(annotated_frame)

            if results:
                detections = extract_detections(
                    results[0],
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                for detection in detections:
                    draw_detection(annotated_frame, detection)

                    previous_center = last_centers_by_track_id.get(
                        detection.track_id)
                    current_center = detection.center
                    current_center_x = current_center[0]

                    if previous_center is not None:
                        crossed_left_line = did_cross_horizontal_line(
                            previous_center=previous_center,
                            current_center=current_center,
                            line_y=left_line_y,
                        )

                        crossed_right_line = did_cross_horizontal_line(
                            previous_center=previous_center,
                            current_center=current_center,
                            line_y=right_line_y,
                        )

                        is_in_left_region = is_inside_x_region(
                            center_x=current_center_x,
                            x_min=LEFT_REGION_X_MIN,
                            x_max=LEFT_REGION_X_MAX,
                        )

                        is_in_right_region = is_inside_x_region(
                            center_x=current_center_x,
                            x_min=RIGHT_REGION_X_MIN,
                            x_max=RIGHT_REGION_X_MAX,
                        )

                        if (
                            crossed_left_line
                            and is_in_left_region
                            and detection.track_id not in counted_on_left_line_ids
                            and not is_duplicate_count_event(
                                recent_events=recent_count_events,
                                current_frame_index=frame_index,
                                line_name="left",
                                class_name=detection.class_name,
                                center=current_center,
                            )
                        ):
                            left_counts[detection.class_name] += 1
                            counted_on_left_line_ids.add(detection.track_id)
                            recent_count_events.append(
                                CountEvent(
                                    frame_index=frame_index,
                                    line_name="left",
                                    class_name=detection.class_name,
                                    center=current_center,
                                )
                            )

                        if (
                            crossed_right_line
                            and is_in_right_region
                            and detection.track_id not in counted_on_right_line_ids
                            and not is_duplicate_count_event(
                                recent_events=recent_count_events,
                                current_frame_index=frame_index,
                                line_name="right",
                                class_name=detection.class_name,
                                center=current_center,
                            )
                        ):
                            right_counts[detection.class_name] += 1
                            counted_on_right_line_ids.add(detection.track_id)
                            recent_count_events.append(
                                CountEvent(
                                    frame_index=frame_index,
                                    line_name="right",
                                    class_name=detection.class_name,
                                    center=current_center,
                                )
                            )

                    last_centers_by_track_id[detection.track_id] = current_center

            draw_counts(annotated_frame, left_counts, right_counts)
            writer.write(annotated_frame)

            if SHOW_WINDOW:
                cv2.imshow(WINDOW_NAME, annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        capture.release()
        writer.release()

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    print_final_counts("Contagem final LEFT FLOW:", left_counts)
    print_final_counts("Contagem final RIGHT FLOW:", right_counts)
    print(f"\nVídeo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
