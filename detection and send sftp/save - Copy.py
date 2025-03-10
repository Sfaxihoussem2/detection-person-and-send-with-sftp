import cv2
from datetime import datetime
import os
import pysftp

# Load class names
classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load model configuration and weights
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(224, 224)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

if __name__ == "__main__":
    cap1 = cv2.VideoCapture(0)
    cap1.set(3, 640)
    cap1.set(4, 480)

    cap2 = cv2.VideoCapture(1)
    cap2.set(3, 640)
    cap2.set(4, 480)

    save_directory = "detected_images"
    os.makedirs(save_directory, exist_ok=True)

    # SFTP credentials
    sftp_host = os.getenv("SFTP_HOST", "Host")
    sftp_port = int(os.getenv("SFTP_PORT", "port"))
    sftp_username = os.getenv("SFTP_USERNAME", "username")
    sftp_password = os.getenv("SFTP_PASSWORD", "pas")

    try:
        with pysftp.Connection(host=sftp_host, port=sftp_port, username=sftp_username, password=sftp_password) as sftp:
            while True:
                success1, img1 = cap1.read()
                success2, img2 = cap2.read()

                result1, objectInfo1 = getObjects(img1, 0.45, 0.2)
                result2, objectInfo2 = getObjects(img2, 0.45, 0.2)

                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename1 = os.path.join(save_directory, f"camera1_{current_datetime}.jpg")
                cv2.imwrite(image_filename1, result1)
                image_filename2 = os.path.join(save_directory, f"camera2_{current_datetime}.jpg")
                cv2.imwrite(image_filename2, result2)

                print(f"Image from camera 1 saved as {image_filename1}")
                print(f"Image from camera 2 saved as {image_filename2}")

                # Upload files to the SFTP server
                sftp.put(image_filename1, f"/transfer-files/camera1_{current_datetime}.jpg")
                sftp.put(image_filename2, f"/transfer-files/camera2_{current_datetime}.jpg")

                cv2.imshow("Output1", result1)
                cv2.imshow("Output2", result2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
