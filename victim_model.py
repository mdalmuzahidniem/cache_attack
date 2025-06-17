import torch
import torch.nn as nn
import ctypes
import os
import time

# Load rdtscp from shared library
lib = ctypes.CDLL("./libtsc.so")
lib.read_tsc.restype = ctypes.c_uint64

def rdtsc():
    return lib.read_tsc()

# === VGG-16 with TSC Logging ===
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
        )

        self.layer_names = [
            "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
            "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
            "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "pool3",
            "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "pool4",
            "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "pool5"
        ]
        self.fc_names = ["fc1", "relu_fc1", "fc2", "relu_fc2", "fc3"]

    def forward(self, x, log_file):
        t_start = rdtsc()
        log_file.write(f"START_TSC {t_start}\n")

        for layer, name in zip(self.conv_layers, self.layer_names):
            t_enter = rdtsc(); log_file.write(f"{t_enter} Entering {name}\n")
            x = layer(x)
            t_leave = rdtsc(); log_file.write(f"{t_leave} Leaving {name}\n")

        x = x.view(x.size(0), -1)

        for layer, name in zip(self.classifier, self.fc_names):
            t_enter = rdtsc(); log_file.write(f"{t_enter} Entering {name}\n")
            x = layer(x)
            t_leave = rdtsc(); log_file.write(f"{t_leave} Leaving {name}\n")

        t_end = rdtsc()
        log_file.write(f"END_TSC {t_end}\n")
        return x

# === Inference Logging ===
if __name__ == "__main__":
    model = VGG16()
    input_tensor = torch.randn(1, 3, 224, 224)

    with open("layer_log.txt", "w") as log_file:
        for i in range(100):
            log_file.write(f"\n=== Inference {i} ===\n")
            _ = model(input_tensor, log_file)
            time.sleep(0.1)

