from queue import Queue
from threading import Thread

import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import numpy as np

import caliscope.logger

# cap = cv2.VideoCapture(0)
from caliscope.packets import PointPacket
from caliscope.tracker import Tracker
from caliscope.trackers.helper import apply_rotation, unrotate_points

logger = caliscope.logger.get(__name__)


class HandTracker(Tracker):
    def __init__(self) -> None:
        self.in_queue = Queue(-1)
        self.out_queue = Queue(-1)

        self.in_queues = {}
        self.out_queues = {}
        self.threads = {}
        self.detectors = {}  # Store detector per port
        self.last_timestamps = {}  # Store last timestamp per port

        # Create tasks directory if it doesn't exist
        self.tasks_dir = "tasks"
        self.model_path = os.path.join(self.tasks_dir, "hand_landmarker.task")
        if not os.path.exists(self.tasks_dir):
            os.makedirs(self.tasks_dir)

        # Download model if it doesn't exist
        if not os.path.exists(self.model_path):
            print("Downloading hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("Model downloaded successfully")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise

        # Initialize HandLandmarker with the downloaded model
        self.base_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.6,
        )

    @property
    def name(self) -> str:
        """Return the name of the tracker for file naming purposes"""
        return "HAND"

    def run_frame_processor(self, port: int, rotation_count: int) -> None:
        # Create detector for this port if it doesn't exist
        if port not in self.detectors:
            self.detectors[port] = vision.HandLandmarker.create_from_options(self.options)
            self.last_timestamps[port] = 0  # Initialize timestamp for this port

        detector = self.detectors[port]

        while True:
            frame, frame_time = self.in_queues[port].get()
            frame = apply_rotation(frame, rotation_count)

            height, width, _ = frame.shape
            # Convert the image format for MediaPipe Tasks
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame)

            # Convert frame_time from seconds to milliseconds
            timestamp_ms = int(frame_time * 1000)
            # Ensure monotonically increasing timestamps
            if timestamp_ms <= self.last_timestamps[port]:
                timestamp_ms = self.last_timestamps[port] + 1
            self.last_timestamps[port] = timestamp_ms

            result = detector.detect_for_video(
                mp_image, int(timestamp_ms)
            )  # TODO: double check frame_time is correct format for the detector

            point_ids = []
            landmark_xy = []

            if result.hand_landmarks:
                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    # Determine if left or right hand
                    hand_label = result.handedness[idx][0].category_name
                    side_adjustment_factor = 0 if hand_label == "Left" else 100

                    for landmark_id, normalized_landmark in enumerate(hand_landmarks):
                        point_ids.append(landmark_id + side_adjustment_factor)

                        # mediapipe expresses in terms of percent of frame, so must map to pixel position
                        x, y = int(normalized_landmark.x * width), int(normalized_landmark.y * height)
                        landmark_xy.append((x, y))

            point_ids = np.array(point_ids)
            landmark_xy = np.array(landmark_xy)
            landmark_xy = unrotate_points(landmark_xy, rotation_count, width, height)

            point_packet = PointPacket(point_ids, landmark_xy)

            self.out_queues[port].put(point_packet)

    def get_points(
        self, frame: np.ndarray, port: int, rotation_count: int, frame_time: float | None = None
    ) -> PointPacket:
        if port not in self.in_queues.keys():
            self.in_queues[port] = Queue(1)
            self.out_queues[port] = Queue(1)

            self.threads[port] = Thread(
                target=self.run_frame_processor,
                args=(port, rotation_count),
                daemon=True,
            )
            self.threads[port].start()

        self.in_queues[port].put((frame, frame_time))
        point_packet = self.out_queues[port].get()

        return point_packet

    def get_point_name(self, point_id: int) -> str:
        return str(point_id)

    def scatter_draw_instructions(self, point_id: int) -> dict:
        if point_id < 100:
            rules = {"radius": 5, "color": (0, 0, 220), "thickness": 3}
        else:
            rules = {"radius": 5, "color": (220, 0, 0), "thickness": 3}
        return rules
