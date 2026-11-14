"""Real-ESRGAN (realesr-general-x4v3, SRVGGNetCompact) を TFLite に変換する。
入力: [1, T, T, 3] float32 (0..1, NHWC)  出力: [1, 4T, 4T, 3]"""
import sys, urllib.request
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TILE = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OUT = sys.argv[2] if len(sys.argv) > 2 else "realesrgan_x4v3.tflite"


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


class NHWC(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y = self.m(x.permute(0, 3, 1, 2))
        return y.clamp(0, 1).permute(0, 2, 3, 1)


url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
urllib.request.urlretrieve(url, "w.pth")
sd = torch.load("w.pth", map_location="cpu")
sd = sd.get("params_ema", sd.get("params", sd))
net = SRVGGNetCompact()
net.load_state_dict(sd, strict=True)
model = NHWC(net).eval()

x = torch.rand(1, TILE, TILE, 3)
with torch.no_grad():
    ref = model(x).numpy()

import ai_edge_torch
edge = ai_edge_torch.convert(model, (x,))
edge.export(OUT)

from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path=OUT)
it.allocate_tensors()
it.set_tensor(it.get_input_details()[0]["index"], x.numpy())
it.invoke()
got = it.get_tensor(it.get_output_details()[0]["index"])
print("in", it.get_input_details()[0]["shape"], "out", got.shape, "max abs diff", float(np.abs(got - ref).max()))
assert np.abs(got - ref).max() < 1e-2
