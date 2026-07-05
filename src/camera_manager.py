import cv2
import threading


class CameraManager:

    def __init__(self):

        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    # ---------------------------------------
    # START CAMERA
    # ---------------------------------------
    def start(self):

        if self.running:
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True

        threading.Thread(
            target=self.update,
            daemon=True
        ).start()

        print("✅ Camera Started")

    # ---------------------------------------
    # UPDATE CAMERA FRAME
    # ---------------------------------------
    def update(self):

        while self.running:

            success, frame = self.cap.read()

            if success:

                frame = cv2.flip(frame, 1)

                with self.lock:
                    self.frame = frame.copy()

    # ---------------------------------------
    # GET CURRENT FRAME
    # ---------------------------------------
    def get_frame(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()

    # ---------------------------------------
    # STOP CAMERA
    # ---------------------------------------
    def stop(self):

        self.running = False

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        print("✅ Camera Stopped")