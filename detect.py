import argparse
import threading
import time
import json
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

# ========================
# Konfigurasi
# ========================
vehicle_labels = {
    'box_truck',
    'double_truck',
    'flat_bed_truck',
    'flatbed_container_truck',
    'pickup_truck',
    'single_truck',
}
odol_label = 'overdimension_load'

# Simpan count total kendaraan
total_counts = {
    'odol': 0,
    'normal': 0
}

# Simpan posisi kendaraan yang sudah dihitung
counted_boxes = []

# Lock untuk thread-safe
lock = threading.Lock()

# Parse argumen
parser = argparse.ArgumentParser()
parser.add_argument('--overlay', type=str, default='box,labels')
args = parser.parse_args()

# Load model
net = detectNet(
    model="model/ssd-mobilenet-odol-30e.onnx",
    labels="model/labels.txt",
    input_blob="input_0",
    output_cvg="scores",
    output_bbox="boxes",
    threshold=0.5
)

# Video source & display
camera = videoSource("csi://0")      # atau "/dev/video0"
display = videoOutput("display://0")

# ========================
# Fungsi bantu
# ========================
def compute_iou(boxA, boxB):
    xA = max(boxA.Left, boxB.Left)
    yA = max(boxA.Top, boxB.Top)
    xB = min(boxA.Right, boxB.Right)
    yB = min(boxA.Bottom, boxB.Bottom)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA.Right - boxA.Left) * (boxA.Bottom - boxA.Top)
    boxBArea = (boxB.Right - boxB.Left) * (boxB.Bottom - boxB.Top)

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def is_new_vehicle(vehicle_box):
    """Cek apakah kendaraan ini sudah pernah dihitung."""
    for prev_box in counted_boxes:
        if compute_iou(vehicle_box, prev_box) > 0.5:  # ambang batas iou
            return False
    return True

# ========================
# Thread Inference
# ========================
def inference_thread():
    global counted_boxes
    while display.IsStreaming():
        img = camera.Capture()
        if img is None:
            continue

        detections = net.Detect(img, overlay=args.overlay)

        vehicles = []
        odols = []

        # Klasifikasi deteksi
        for det in detections:
            label = net.GetClassDesc(det.ClassID)
            if label == odol_label:
                odols.append(det)
            elif label in vehicle_labels:
                vehicles.append((det, label))

        # Proses kendaraan
        with lock:
            for det, label in vehicles:
                if is_new_vehicle(det):
                    # Cek apakah tergolong ODOL (beririsan dengan muatan)
                    is_odol = any(compute_iou(det, od) > 0.3 for od in odols)
                    if is_odol:
                        total_counts['odol'] += 1
                    else:
                        total_counts['normal'] += 1

                    # Simpan posisi untuk hindari double count
                    counted_boxes.append(det)

        display.Render(img)
        display.SetStatus(f"ODOL: {total_counts['odol']} | Normal: {total_counts['normal']} | {net.GetNetworkFPS():.0f} FPS")

# ========================
# Thread Print Periodik (boleh dihapus kalau nggak perlu)
# ========================
def print_result_thread():
    while True:
        time.sleep(5)
        with lock:
            print("==== TOTAL COUNT ====")
            print(json.dumps(total_counts, indent=2))

# ========================
# Start Threads
# ========================
threading.Thread(target=inference_thread, daemon=True).start()
threading.Thread(target=print_result_thread, daemon=True).start()

# Main thread: tunggu display selesai
while display.IsStreaming():
    time.sleep(1)
