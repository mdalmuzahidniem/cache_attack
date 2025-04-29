
import torch
import torch.nn as nn
import time

# High-resolution clock for alignment
def now_ns():
    return time.monotonic_ns()

# === VGG-16 with Entering/Leaving Logging ===
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 64, 3, padding=1),    # conv1_1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),   # conv1_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # pool1

            nn.Conv2d(64, 128, 3, padding=1),  # conv2_1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), # conv2_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # pool2

            nn.Conv2d(128, 256, 3, padding=1), # conv3_1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), # conv3_2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), # conv3_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # pool3

            nn.Conv2d(256, 512, 3, padding=1), # conv4_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), # conv4_2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), # conv4_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # pool4

            nn.Conv2d(512, 512, 3, padding=1), # conv5_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), # conv5_2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), # conv5_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)                 # pool5
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # fc1
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), # fc2
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)    # fc3
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
        for layer, name in zip(self.conv_layers, self.layer_names):
            t_enter = now_ns(); log_file.write(f"{t_enter} Entering {name}\n")
            x = layer(x)
            t_leave = now_ns(); log_file.write(f"{t_leave} Leaving {name}\n")

        x = x.view(x.size(0), -1)

        for layer, name in zip(self.classifier, self.fc_names):
            t_enter = now_ns(); log_file.write(f"{t_enter} Entering {name}\n")
            x = layer(x)
            t_leave = now_ns(); log_file.write(f"{t_leave} Leaving {name}\n")

        t = now_ns(); log_file.write(f"{t} Finished forward\n")
        return x

# === Inference Logging ===
if __name__ == "__main__":
    model = VGG16()
    input_tensor = torch.randn(1, 3, 224, 224)  # match VGG-16 and Cache Telepathy

    with open("layer_log.txt", "w") as log_file:
        for i in range(100):
            log_file.write(f"\n=== Inference {i} ===\n")
            _ = model(input_tensor, log_file)
            time.sleep(0.1)
