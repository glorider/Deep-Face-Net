import cv2
import insightface
import threading

from core.config import ANALYSIS_MODEL
from download_models import check_model_status

FACE_ANALYSER_ = None
LOCK_ = threading.Lock()


def get_face_analyser():
    global FACE_ANALYSER_

    if FACE_ANALYSER_ is None:
        with LOCK_:
            if FACE_ANALYSER_ is None:
                # Check if the analyser model (buffalo_l) is downloaded first
                is_downloaded, _, _ = check_model_status(ANALYSIS_MODEL)
                if not is_downloaded:
                    raise Exception(f"The '{ANALYSIS_MODEL}' model is not loaded/downloaded. Please go to the Models tab to download it first.")

                analyser = insightface.app.FaceAnalysis(name=ANALYSIS_MODEL)
                analyser.prepare(ctx_id=0, det_size=(640, 640))
                FACE_ANALYSER_ = analyser

    return FACE_ANALYSER_
