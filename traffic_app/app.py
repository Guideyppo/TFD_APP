import os
import torch
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ------------------------
# โหลด YOLOv8 จาก pretrained
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov8n.pt')

# label ของ traffic light (COCO dataset มี class: 'traffic light')
labels = ["traffic light"]

# preprocessing (ยังไม่ใช้โดยตรง แต่เผื่ออนาคต)
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# ------------------------
# Flask web app
# ------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # บันทึกไฟล์ที่อัพโหลด
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # ใช้ YOLOv8 ตรวจจับ
        results = model(filepath)
        result = results[0]

        # plot() คืนเป็น numpy array ของภาพที่มีกรอบวัตถุ
        img_with_boxes = result.plot()

        # save ภาพที่มีกรอบวัตถุ
        img_save = Image.fromarray(img_with_boxes)
        result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_" + file.filename)
        img_save.save(result_path)

        # ตรวจสอบว่ามี traffic light หรือไม่ + confidence
        found_traffic_light = False
        conf_score = 0.0

        if result.boxes and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i])
                class_name = model.names[cls_id] if hasattr(model, 'names') and cls_id < len(model.names) else str(cls_id)

                if class_name in labels:
                    found_traffic_light = True
                    conf_score = float(result.boxes.conf[i].item())  # ดึงค่า confidence
                    break

        # สร้างข้อความผลลัพธ์
        result_text = "ตรวจพบไฟจราจร" if found_traffic_light else "ไม่พบไฟจราจร"
        conf_text = f"{conf_score*100:.1f} %" if found_traffic_light else "0.0 %"

        return render_template("index.html",
                               filename="result_" + file.filename,
                               result=result_text,
                               conf=conf_text)

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return url_for("static", filename="uploads/" + filename)

if __name__ == "__main__":
    app.run(debug=True)