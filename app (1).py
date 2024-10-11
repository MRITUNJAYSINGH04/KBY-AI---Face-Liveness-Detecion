import sys
sys.path.append('.')

import os
import numpy as np
import base64
import io

from PIL import Image, ExifTags
from flask import Flask, request, jsonify
from flask_cors import CORS
from facesdk import getMachineCode
from facesdk import setActivation
from facesdk import faceDetection
from facesdk import initSDK
from facebox import FaceBox

licensePath = "license.txt"
license = ""

# Get a specific environment variable by name
license = os.environ.get("LICENSE")

# Check if the variable exists
if license is not None:
    print("Value of LICENSE:", license)
else:
    license = ""
    try:
        with open(licensePath, 'r') as file:
            license = file.read().strip()
    except IOError as exc:
        print("failed to open license.txt: ", exc.errno)
    print("license: ", license)

livenessThreshold = 0.7
yawThreshold = 10
pitchThreshold = 10
rollThreshold = 10
occlusionThreshold = 0.9
eyeClosureThreshold = 0.8
mouthOpeningThreshold = 0.5
borderRate = 0.05
smallFaceThreshold = 100
lowQualityThreshold = 0.3
hightQualityThreshold = 0.7
luminanceDarkThreshold = 50
luminanceLightThreshold = 200

maxFaceCount = 10

machineCode = getMachineCode()
print("machineCode: ", machineCode.decode('utf-8'))

ret = setActivation(license.encode('utf-8'))
print("activation: ", ret)

ret = initSDK("data".encode('utf-8'))
print("init: ", ret)

app = Flask(__name__) 
CORS(app)

def apply_exif_rotation(image):
    # Get the EXIF data
    try:
        exif = image._getexif()
        if exif is not None:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break

            # Get the orientation value
            orientation = exif.get(orientation, None)

            # Apply the appropriate rotation based on the orientation
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

    except AttributeError:
        print("No EXIF data found")

    return image
    
@app.route('/check_liveness', methods=['POST'])
def check_liveness():
    faces = []
    isNotFront = None
    isOcclusion = None
    isEyeClosure = None
    isMouthOpening = None
    isBoundary = None
    isSmall = None
    quality = None
    luminance = None
    livenessScore = None

    file = request.files['file']

    try:
        image = apply_exif_rotation(Image.open(file)).convert('RGB')
    except:
        result = "Failed to open file"
        faceState = {"is_not_front": isNotFront, "is_occluded": isOcclusion, "eye_closed": isEyeClosure, "mouth_opened": isMouthOpening, 
                        "is_boundary_face": isBoundary, "is_small": isSmall, "quality": quality, "luminance": luminance, "result": result, "liveness_score": livenessScore}
        response = jsonify({"face_state": faceState, "faces": faces})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response


    image_np = np.asarray(image)

    faceBoxes = (FaceBox * maxFaceCount)()
    faceCount = faceDetection(image_np, image_np.shape[1], image_np.shape[0], faceBoxes, maxFaceCount)       

    for i in range(faceCount):
        landmark_68 = []
        for j in range(68):
            landmark_68.append({"x": faceBoxes[i].landmark_68[j * 2], "y": faceBoxes[i].landmark_68[j * 2 + 1]})
        faces.append({"x1": faceBoxes[i].x1, "y1": faceBoxes[i].y1, "x2": faceBoxes[i].x2, "y2": faceBoxes[i].y2, 
                      "liveness": faceBoxes[i].liveness, 
                      "yaw": faceBoxes[i].yaw, "roll": faceBoxes[i].roll, "pitch": faceBoxes[i].pitch,
                      "face_quality": faceBoxes[i].face_quality, "face_luminance": faceBoxes[i].face_luminance, "eye_dist": faceBoxes[i].eye_dist,
                      "left_eye_closed": faceBoxes[i].left_eye_closed, "right_eye_closed": faceBoxes[i].right_eye_closed,
                      "face_occlusion": faceBoxes[i].face_occlusion, "mouth_opened": faceBoxes[i].mouth_opened,
                      "landmark_68": landmark_68})
    
    result = ""
    if faceCount == 0:
        result = "No face"
#    elif faceCount > 1:
#        result = "Multiple face"
    elif faceCount < 0:
        result = "License error!"
    else:
        livenessScore = faceBoxes[0].liveness
        if livenessScore > livenessThreshold:
            result = "Real"
        else:
            result = "Spoof"
        
        isNotFront = True
        isOcclusion = False
        isEyeClosure = False
        isMouthOpening = False
        isBoundary = False
        isSmall = False
        quality = "Low"
        luminance = "Dark"
        if abs(faceBoxes[0].yaw) < yawThreshold and abs(faceBoxes[0].roll) < rollThreshold and abs(faceBoxes[0].pitch) < pitchThreshold:
            isNotFront = False
        
        if faceBoxes[0].face_occlusion > occlusionThreshold:
            isOcclusion = True

        if faceBoxes[0].left_eye_closed > eyeClosureThreshold or faceBoxes[0].right_eye_closed > eyeClosureThreshold:
            isEyeClosure = True

        if faceBoxes[0].mouth_opened > mouthOpeningThreshold:
            isMouthOpening = True
        
        if (faceBoxes[0].x1 < image_np.shape[1] * borderRate or 
            faceBoxes[0].y1 < image_np.shape[0] * borderRate or 
                faceBoxes[0].x1 > image_np.shape[1] - image_np.shape[1] * borderRate or 
                faceBoxes[0].x1 > image_np.shape[0] - image_np.shape[0] * borderRate):
            isBoundary = True

        if faceBoxes[0].eye_dist < smallFaceThreshold:
            isSmall = True
        
        if faceBoxes[0].face_quality < lowQualityThreshold:
            quality = "Low"
        elif faceBoxes[0].face_quality < hightQualityThreshold:
            quality = "Medium"
        else:
            quality = "High"
        
        if faceBoxes[0].face_luminance < luminanceDarkThreshold:
            luminance = "Dark"
        elif faceBoxes[0].face_luminance < luminanceLightThreshold:
            luminance = "Normal"
        else:
            luminance = "Light"

    faceState = {"is_not_front": isNotFront, "is_occluded": isOcclusion, "eye_closed": isEyeClosure, "mouth_opened": isMouthOpening, 
                    "is_boundary_face": isBoundary, "is_small": isSmall, "quality": quality, "luminance": luminance, "result": result, "liveness_score": livenessScore}
    response = jsonify({"face_state": faceState, "faces": faces})

    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

@app.route('/check_liveness_base64', methods=['POST'])
def check_liveness_base64():
    faces = []
    isNotFront = None
    isOcclusion = None
    isEyeClosure = None
    isMouthOpening = None
    isBoundary = None
    isSmall = None
    quality = None
    luminance = None
    livenessScore = None

    content = request.get_json()

    try:
        imageBase64 = content['base64']
        image_data = base64.b64decode(imageBase64)
        image = apply_exif_rotation(Image.open(io.BytesIO(image_data))).convert("RGB")
    except:
        result = "Failed to open file"
        faceState = {"is_not_front": isNotFront, "is_occluded": isOcclusion, "eye_closed": isEyeClosure, "mouth_opened": isMouthOpening, 
                        "is_boundary_face": isBoundary, "is_small": isSmall, "quality": quality, "luminance": luminance, "result": result, "liveness_score": livenessScore}
        response = jsonify({"face_state": faceState, "faces": faces})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response


    image_np = np.asarray(image)

    faceBoxes = (FaceBox * maxFaceCount)()
    faceCount = faceDetection(image_np, image_np.shape[1], image_np.shape[0], faceBoxes, maxFaceCount)

    for i in range(faceCount):
        landmark_68 = []
        for j in range(68):
            landmark_68.append({"x": faceBoxes[i].landmark_68[j * 2], "y": faceBoxes[i].landmark_68[j * 2 + 1]})
        faces.append({"x1": faceBoxes[i].x1, "y1": faceBoxes[i].y1, "x2": faceBoxes[i].x2, "y2": faceBoxes[i].y2, 
                      "liveness": faceBoxes[i].liveness, 
                      "yaw": faceBoxes[i].yaw, "roll": faceBoxes[i].roll, "pitch": faceBoxes[i].pitch,
                      "face_quality": faceBoxes[i].face_quality, "face_luminance": faceBoxes[i].face_luminance, "eye_dist": faceBoxes[i].eye_dist,
                      "left_eye_closed": faceBoxes[i].left_eye_closed, "right_eye_closed": faceBoxes[i].right_eye_closed,
                      "face_occlusion": faceBoxes[i].face_occlusion, "mouth_opened": faceBoxes[i].mouth_opened,
                      "landmark_68": landmark_68})

    result = ""
    if faceCount == 0:
        result = "No face"
#    elif faceCount > 1:
#        result = "Multiple face"
    elif faceCount < 0:
        result = "License error!"
    else:
        livenessScore = faceBoxes[0].liveness
        if livenessScore > livenessThreshold:
            result = "Real"
        else:
            result = "Spoof"
        
        isNotFront = True
        isOcclusion = False
        isEyeClosure = False
        isMouthOpening = False
        isBoundary = False
        isSmall = False
        quality = "Low"
        luminance = "Dark"
        if abs(faceBoxes[0].yaw) < yawThreshold and abs(faceBoxes[0].roll) < rollThreshold and abs(faceBoxes[0].pitch) < pitchThreshold:
            isNotFront = False
        
        if faceBoxes[0].face_occlusion > occlusionThreshold:
            isOcclusion = True

        if faceBoxes[0].left_eye_closed > eyeClosureThreshold or faceBoxes[0].right_eye_closed > eyeClosureThreshold:
            isEyeClosure = True

        if faceBoxes[0].mouth_opened > mouthOpeningThreshold:
            isMouthOpening = True
        
        if (faceBoxes[0].x1 < image_np.shape[1] * borderRate or 
            faceBoxes[0].y1 < image_np.shape[0] * borderRate or 
                faceBoxes[0].x1 > image_np.shape[1] - image_np.shape[1] * borderRate or 
                faceBoxes[0].x1 > image_np.shape[0] - image_np.shape[0] * borderRate):
            isBoundary = True

        if faceBoxes[0].eye_dist < smallFaceThreshold:
            isSmall = True
        
        if faceBoxes[0].face_quality < lowQualityThreshold:
            quality = "Low"
        elif faceBoxes[0].face_quality < hightQualityThreshold:
            quality = "Medium"
        else:
            quality = "High"
        
        if faceBoxes[0].face_luminance < luminanceDarkThreshold:
            luminance = "Dark"
        elif faceBoxes[0].face_luminance < luminanceLightThreshold:
            luminance = "Normal"
        else:
            luminance = "Light"

    faceState = {"is_not_front": isNotFront, "is_occluded": isOcclusion, "eye_closed": isEyeClosure, "mouth_opened": isMouthOpening, 
                    "is_boundary_face": isBoundary, "is_small": isSmall, "quality": quality, "luminance": luminance, "result": result, "liveness_score": livenessScore}
    response = jsonify({"face_state": faceState, "faces": faces})

    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
