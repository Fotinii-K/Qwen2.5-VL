import pyzed.sl as sl
class Wrapper:
    """
    Wrapper for ZED API (v3.6) with settings loaded from Hydra YAML.
    """
    def __init__(self, camera_cfg: dict):
        """
        Initializes the ZED camera with settings from Hydra YAML.
        =INPUT=
            camera_cfg - Dict from Hydra (cfg.camera)
        """
        self.camera_cfg = camera_cfg
        self.initial_parameters = sl.InitParameters()
        self.runtime_parameters = sl.RuntimeParameters()
        self.streaming_parameters = sl.StreamingParameters()
        self.recording_parameters = sl.RecordingParameters()
        self.camera = sl.Camera()
        self.mat_image = sl.Mat()
        self.mat_measure = sl.Mat()
        self.grab_status = None
        self.timestamp_nanoseconds = 0
        # Set camera input type
        if camera_cfg["connection_type"] == "id":
            self.initial_parameters.set_from_camera_id(camera_cfg.get("id", 0))
        elif camera_cfg["connection_type"] == "serial":
            self.initial_parameters.set_from_serial_number(camera_cfg["serial_number"])
        elif camera_cfg["connection_type"] == "svo":
            self.initial_parameters.set_from_svo_file(camera_cfg["svo_input_filename"])
        elif camera_cfg["connection_type"] == "stream":
            self.initial_parameters.set_from_stream(camera_cfg["sender_ip"], camera_cfg["port"])
        # Set additional camera parameters
        self.initial_parameters.depth_mode = getattr(sl.DEPTH_MODE, camera_cfg["depth_mode"])
        self.initial_parameters.camera_resolution = getattr(sl.RESOLUTION, camera_cfg["resolution"])
        self.initial_parameters.coordinate_units = getattr(sl.UNIT, camera_cfg["coordinate_units"])
        self.initial_parameters.camera_fps = camera_cfg["camera_fps"]
        self.initial_parameters.depth_minimum_distance = camera_cfg["depth_min_distance"]
        self.initial_parameters.depth_maximum_distance = camera_cfg["depth_max_distance"]
        # Set runtime parameters
        self.runtime_parameters.enable_fill_mode = camera_cfg["enable_fill_mode"]
        # Numpy output arrays
        self.output_image = None
        self.output_measure = None
    
    def get_intrinsic(self):
        """
        Get camera intrinsic parameters.
        =OUTPUT=
            intrinsic_left, intrinsic_right - CameraParameters (see Stereolabs API)
        """
        camera_information = self.camera.get_camera_information()
        intrinsic_left = camera_information.camera_configuration.calibration_parameters_raw.left_cam
        intrinsic_right = camera_information.camera_configuration.calibration_parameters_raw.right_cam
        return intrinsic_left, intrinsic_right
    
    def open_input_source(self):
        """
        Open the camera, stream, or file connection.
        """
        status = self.camera.open(self.initial_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Failed to open input source '{self.camera_cfg['connection_type']}': {status}")
        return True
    def close_input_source(self):
        """
        Free mat memory (CPU + GPU) and close the camera connection.
        """
        self.output_image = None
        self.output_measure = None
        self.mat_image.free(sl.MEM.CPU)
        self.mat_measure.free(sl.MEM.CPU)
        self.camera.close()
        return True
    
    def start_stream(self):
        """
        Start streaming data. Must be called after open_input_source.
        """
        if not self.camera.is_opened():
            raise Exception("Camera is not opened. Cannot start streaming.")
        status = self.camera.enable_streaming(self.streaming_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Failed to start stream: {status}")
        return True
    
    def stop_stream(self):
        """
        Stop streaming.
        """
        self.camera.disable_streaming()
        return True
    
    def start_recording(self, filename="output.svo"):
        """
        Start recording data into an SVO file.
        """
        if not self.camera.is_opened():
            raise Exception("Camera is not opened. Cannot start recording.")
        self.recording_parameters.video_filename = filename
        status = self.camera.enable_recording(self.recording_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Failed to start recording: {status}")
        return True
    
    def stop_recording(self):
        """
        Stop recording.
        """
        self.camera.disable_recording()
        return True
    def retrieve(self, is_image=True, is_measure=False):
        """
        Retrieve captured image frame or depth measure.
        =NOTES=
            - It is likely more efficient to obtain SIDE_BY_SIDE and then split the array
              rather than calling retrieve_image twice for left and right separately.
            - However, obtaining only IMAGE_LEFT is likely more efficient than getting SIDE_BY_SIDE.
        TODO: Optimize by using GPU and reducing unnecessary CPU-GPU copies.
        """
        self.grab_status = self.camera.grab(self.runtime_parameters)
        if self.grab_status == sl.ERROR_CODE.SUCCESS:
            if is_image:
                self.camera.retrieve_image(self.mat_image, sl.VIEW.LEFT)
                self.output_image = self.mat_image.get_data()
                self.timestamp_nanoseconds = self.mat_image.timestamp.data_ns
            if is_measure:
                self.camera.retrieve_measure(self.mat_measure, sl.MEASURE.DEPTH_U16_MM)
                self.output_measure = self.mat_measure.get_data()
        else:
            print("Failed to grab camera frame.")
            return False
        return True
    
    def export_png(self, filename, is_image=True, is_measure=False):
        """
        Export the last retrieved frame as a PNG image.
        =INPUT=
            filename - string
                Path name + file name + .png extension
        """
        if self.grab_status != sl.ERROR_CODE.SUCCESS:
            return False
        if is_image:
            self.mat_image.write(filename)
        if is_measure:
            self.mat_measure.write(filename)
        return True
    
    def depth_at_xy(self, xpos: int = 0, ypos: int = 0):
        """
        Get the depth value at a specific pixel coordinate.
        =INPUT=
            xpos - int : X-coordinate of the pixel
            ypos - int : Y-coordinate of the pixel
        =OUTPUT=
            Depth value at (xpos, ypos)
        """
        if self.output_measure is not None:
            return self.output_measure[xpos, ypos]
        return None