import sys
from pathlib import Path

import toml
from PyQt6.QtWidgets import QApplication

from pyxyfy import __root__
from pyxyfy.cameras.camera_array import CameraArray
from pyxyfy.cameras.camera_array_builder import CameraArrayBuilder
from pyxyfy.gui.vizualize.capture_volume_visualizer import \
    CaptureVolumeVisualizer
from pyxyfy.calibration.omnicalibrator import OmniCalibrator

# session_path  = Path(__root__, "tests", "3_cameras_triangular")
# session_path  = Path(__root__, "tests", "3_cameras_middle")
# session_path  = Path(__root__, "tests", "3_cameras_midlinear")
# session_path  = Path(__root__, "tests", "3_cameras_linear")
session_path  = Path(__root__, "tests", "tripod2")

config_path  = Path(session_path,"config.toml")


point_data_path = Path(session_path, "point_data.csv")

omnical = OmniCalibrator(
    config_path,
    point_data_path,
)

# omnical.stereo_calibrate_all(boards_sampled=5)

array_builder = CameraArrayBuilder(config_path)
camera_array:CameraArray = array_builder.get_camera_array()


app = QApplication(sys.argv)

vizr = CaptureVolumeVisualizer(camera_array=camera_array)

sys.exit(app.exec())
