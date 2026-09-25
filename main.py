from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

HTML_PATH = BASE_DIR / "objection-detection.html"

MODEL_PATH = BASE_DIR / "best.pt"


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Screw Defect Detection"
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("\n" + "=" * 50)
print("LOADING SCREW DETECTION MODEL")
print("=" * 50)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"best.pt not found at: {MODEL_PATH}"
    )


model = YOLO(str(MODEL_PATH))


print("Model loaded successfully.")
print("Classes:", model.names)

print("=" * 50)


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/")
async def home():

    return FileResponse(
        HTML_PATH
    )


# =========================================================
# SERVER STATUS
# =========================================================

@app.get("/status")
async def status():

    return {
        "server": "online",
        "model": "loaded",
        "classes": model.names
    }


# =========================================================
# SCREW DETECTION
# =========================================================

@app.post("/compare")
async def detect_screw(
    file: UploadFile = File(...)
):

    try:

        # -------------------------------------------------
        # READ IMAGE FROM CAMERA
        # -------------------------------------------------

        image_bytes = await file.read()


        # -------------------------------------------------
        # CONVERT BYTES TO OPENCV IMAGE
        # -------------------------------------------------

        frame = cv2.imdecode(
            np.frombuffer(
                image_bytes,
                np.uint8
            ),
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return {
                "success": False,
                "detected": False,
                "result": "INVALID IMAGE",
                "message": "Image could not be read."
            }


        # -------------------------------------------------
        # YOLO DETECTION
        # -------------------------------------------------

        results = model(
            frame,
            conf=0.35,
            verbose=False
        )


        detections = []


        # -------------------------------------------------
        # PROCESS YOLO RESULTS
        # -------------------------------------------------

        for result in results:


            if result.boxes is None:

                continue


            for box in result.boxes:


                # -----------------------------------------
                # BOUNDING BOX
                # -----------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )


                # -----------------------------------------
                # CONFIDENCE
                # -----------------------------------------

                confidence = float(
                    box.conf[0]
                )


                # -----------------------------------------
                # CLASS ID
                # -----------------------------------------

                class_id = int(
                    box.cls[0]
                )


                # -----------------------------------------
                # CLASS NAME
                # -----------------------------------------

                class_name = model.names[
                    class_id
                ]


                # -----------------------------------------
                # GOOD / DEFECTIVE
                # -----------------------------------------

                if class_name.lower() == "ok":

                    status = "GOOD"

                    final_result = "GOOD SCREW"


                else:

                    status = "DEFECTIVE"

                    final_result = "DEFECTIVE SCREW"


                # -----------------------------------------
                # SAVE DETECTION
                # -----------------------------------------

                detections.append({

                    "class": class_name,

                    "confidence": round(
                        confidence * 100,
                        2
                    ),

                    "status": status,

                    "result": final_result,

                    "box": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]

                })


        # =================================================
        # NO SCREW DETECTED
        # =================================================

        if len(detections) == 0:

            return {

                "success": True,

                "detected": False,

                "result": "NO SCREW DETECTED",

                "message":
                    "Screw camera ke saamne lao."

            }


        # =================================================
        # SELECT BEST DETECTION
        # =================================================

        best_detection = max(

            detections,

            key=lambda x:
                x["confidence"]

        )


        # =================================================
        # SEND RESULT TO FRONTEND
        # =================================================

        return {

            "success": True,

            "detected": True,

            "result":
                best_detection["result"],

            "status":
                best_detection["status"],

            "class":
                best_detection["class"],

            "confidence":
                best_detection["confidence"],

            "box":
                best_detection["box"],

            "detections":
                detections

        }


    except Exception as e:


        print(
            "Detection error:",
            str(e)
        )


        return {

            "success": False,

            "detected": False,

            "result": "ERROR",

            "message": str(e)

        }


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def startup_event():

    print("\n" + "=" * 50)

    print(
        "SCREW DEFECT DETECTION SERVER"
    )

    print("=" * 50)

    print(
        "Server is ready."
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Classes:",
        model.names
    )

    print("=" * 50)