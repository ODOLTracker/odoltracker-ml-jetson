import argparse
from config import config
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

# Argument parser untuk overlay
parser = argparse.ArgumentParser(description="DetectNet ODOL Counting")
parser.add_argument('--overlay', type=str, default='box,labels', 
                    help="Set overlay options (e.g., 'box,labels' or 'none')")
args = parser.parse_args()

# Load model custom
net = detectNet(
    model="model/ssd-mobilenet-odol-30e.onnx",
    labels="model/labels.txt",
    input_blob="input_0",
    output_cvg="scores",
    output_bbox="boxes",
    threshold=0.5,
)

# Sumber video
camera = videoSource("csi://0")      # Ganti sesuai sumber kamera kamu
display = videoOutput("display://0") # Ganti sesuai output yang kamu inginkan

# Fungsi menghitung IoU
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

# Inisialisasi total count
total_odol = 0
total_normal = 0
vehicle_ids_seen = set()  # Untuk hindari double count (jika pakai ID unik)

# Main loop
while display.IsStreaming():
    img = camera.Capture()
    if img is None:
        continue

    detections = net.Detect(img, overlay=args.overlay)

    vehicles = []
    odols = []

    # Pisahkan deteksi berdasarkan label
    for det in detections:
        label = net.GetClassDesc(det.ClassID)
        if label in config.vehicle_labels:
            vehicles.append((det, label))
        elif label == config.odol_label:
            odols.append(det)

    # Cek dan hitung
    for vehicle, label in vehicles:
        is_odol = False
        for od in odols:
            iou = compute_iou(vehicle, od)
            if iou > 0.3:
                is_odol = True
                break
        if is_odol:
            total_odol += 1
        else:
            total_normal += 1

    # Render saja, tapi tidak print per frame
    display.Render(img)
    display.SetStatus("ODOL: {} | Normal: {} | {:.0f} FPS".format(total_odol, total_normal, net.GetNetworkFPS()))

# Setelah selesai (misal kamu stop pakai Ctrl+C), cetak total
print("=== TOTAL HASIL DETEKSI ===")
print(f"Total Kendaraan ODOL   : {total_odol}")
print(f"Total Kendaraan Normal : {total_normal}")