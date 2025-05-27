# Anime4K-PyTorch:
#   https://colab.research.google.com/drive/11xAn4fyAUJPZOjrxwnL2ipl_1DGGegkB#scrollTo=KVBLI1M20vWQ
#
# based on Anime4K:
#   https://github.com/bloc97/Anime4K
#
# Used under the MIT License.

import os
import re
import math

# import locale
# locale.getpreferredencoding = lambda: "UTF-8"

import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

to_pil = torchvision.transforms.ToPILImage()
to_tensor = torchvision.transforms.ToTensor()

def conv_layer(in_channels, out_channels, kernel_size):
    padding = int((kernel_size - 1) / 2)
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return F.relu(torch.cat((x, -x), 1))

class anime4k(nn.Module):
    def __init__(self, block_depth=7, stack_list=5, num_feat=12, last=False, scale=2, single_tail=False, upscale_mode="bilinear"):
        """
        `block_depth`: Num of hidden layer shaders + 1, must be >= num_feat
        `num_feat`: Total output channel of input Conv2D shaders
        `stack_list` = Num of binds of last_conv2d / 2
        `last`: last_conv2d's kernel=3 if True else 1
        `single_tail`: Set this to True if there is only 1 conv2d_last shader (e.g. Upscale_S, Upscale_M)

        Input conv2d shaders are first shaders having "MAIN_texOff", output conv2d shaders are last ones having "conv2d_last_tf".

        Credit:
            https://github.com/kato-megumi
        """
        super(anime4k, self).__init__()
        self.act = CReLU()
        if type(stack_list) == int:
            stack_list = list(range(-stack_list, 0))
        self.stack_list = stack_list
        self.scale = scale
        self.ps = nn.PixelShuffle(self.scale)
        self.conv_head = conv_layer(3, num_feat, kernel_size=3)
        self.conv_mid = nn.ModuleList(
            [
                conv_layer(num_feat * 2, num_feat, kernel_size=3)
                for _ in range(block_depth - 1)
            ]
        )
        tail_out_c = 4 if single_tail else 3 * scale * scale
        if last:
            self.conv_tail = conv_layer(2 * num_feat * len(stack_list), tail_out_c, kernel_size=3)
        else:
            self.conv_tail = conv_layer(2 * num_feat * len(stack_list), tail_out_c, kernel_size=1)
        self.upscale_mode = upscale_mode

    def forward(self, x):
        out = self.act(self.conv_head(x))
        depth_list = [out]
        for conv in self.conv_mid:
            out = self.act(conv(out))
            depth_list.append(out)
        out = self.conv_tail(torch.cat([depth_list[i] for i in self.stack_list], 1))
        if self.scale != 1:
            out = self.ps(out) + F.interpolate(x, scale_factor=self.scale, mode=self.upscale_mode)
        else:
            out += x
        return torch.clamp(out, max=1.0, min=0.0)

    def import_param(self, filename):
        for param in self.parameters():
            param.requires_grad = False
        with open(filename) as f:
            text = f.read()
        pattern = r'-?\d+(\.\d{4,})(e-?\d+)?'
        iter = re.finditer(pattern, text)
        convert(self.conv_head, iter)
        for conv in self.conv_mid:
            convert(conv, iter)
        convert(self.conv_tail, iter, True)
        check = next(iter, None)
        if check is None:
            print("pass")
        else:
            print("---failed---\n", check)


def convert(c, iter, doswap=False):
    swap = [0, 2, 1, 3]
    out_chan, in_chan, width, height = c.weight.shape
    for to in range(math.ceil(out_chan / 4)):
        for ti in range(math.ceil(in_chan / 4)):
            for w in range(width):
                for h in range(height):
                    for i in range(min(4, in_chan)):
                        for o in range(min(4, out_chan)):
                            o = swap[o] if doswap else o
                            c.weight.data[to * 4 + o, ti * 4 + i, w, h] = float(next(iter).group(0))
        for o in range(min(4, out_chan)):
            o = swap[o] if doswap else o
            c.bias.data[to * 4 + o] = float(next(iter).group(0))

def get_luma(x):
    x = x[:, 0] * 0.299 + x[:, 1] * 0.587 + x[:, 2] * 0.114
    x = x.unsqueeze(1)
    return x

class MaxPoolKeepShape(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPoolKeepShape, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        kernel_height, kernel_width = self.kernel_size
        pad_height = (((height - 1) // self.stride + 1) - 1) * self.stride + kernel_height - height
        pad_width = (((width - 1) // self.stride + 1) - 1) * self.stride + kernel_width - width

        x = F.pad(x, (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2))
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x

# Ref: https://github.com/bloc97/Anime4K/blob/master/glsl/Restore/Anime4K_Clamp_Highlights.glsl
class ClampHighlight(nn.Module):
    def __init__(self):
        super(ClampHighlight, self).__init__()
        self.max_pool = MaxPoolKeepShape(kernel_size=(5, 5), stride=1)
    def forward(self, shader_img, orig_img):
        curr_luma = get_luma(shader_img)
        statsmax = self.max_pool(get_luma(orig_img))
        if statsmax.shape != curr_luma.shape:
            statsmax = F.interpolate(statsmax, curr_luma.shape[2:4])
        new_luma = torch.min(curr_luma, statsmax)
        return shader_img - (curr_luma - new_luma)

class AutoDownscalePre(nn.Module):
    def __init__(self, factor, lower_thresh=1.2, upper_thresh=2.0, upscale_mode="bilinear"):
        super(AutoDownscalePre, self).__init__()
        self.factor = factor // 2
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh
        self.upscale_mode = upscale_mode
    def forward(self, x, screen_width=1920, screen_height=1080):
        h, w = x.shape[2:]
        # RPN expression is so weird to understand. Let assume that ChatGPT is right
        # https://github.com/bloc97/Anime4K/blob/master/glsl/Upscale/Anime4K_AutoDownscalePre_x2.glsl#L30
        # https://github.com/bloc97/Anime4K/blob/master/glsl/Upscale/Anime4K_AutoDownscalePre_x4.glsl#L30
        h_ratio = h / screen_height / self.factor
        w_ratio = w / screen_width / self.factor
        if (h_ratio > self.lower_thresh) and \
           (w_ratio > self.lower_thresh) and \
           (h_ratio < self.upper_thresh) and \
           (w_ratio < self.upper_thresh):
            return F.interpolate(x, (screen_height // self.factor, screen_width // self.factor), mode=self.upscale_mode)
        return x

class Anime4KPipeline(nn.Module):
    def __init__(self, *models, screen_width=1920, screen_height=1080, final_stage_upscale_mode="bilinear"):
        super(Anime4KPipeline, self).__init__()
        self.models = list(models)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.final_stage_upscale_mode = final_stage_upscale_mode
    def to(self, device_or_dtype):
        for i in range(len(self.models)):
            self.models[i] = self.models[i].to(device_or_dtype)
        return self
    def half(self):
        return self.to(torch.half)
    def forward(self, x):
        clamp_hightlight, orig_img = None, None
        for model in self.models:
            if model.__class__.__name__ == "AutoDownscalePre":
                x = model(x, self.screen_width, self.screen_height)
                continue
            if model.__class__.__name__ == "ClampHighlight":
                clamp_hightlight = model
                orig_img = x.clone()
                continue
            x = model(x)
        if clamp_hightlight is not None:
            x = clamp_hightlight(x, orig_img)
        x = F.interpolate(x, (self.screen_height, self.screen_width), mode=self.final_stage_upscale_mode)
        return x

model_dict = {
    "Upscale_S": ("Anime4K_Upscale_CNN_x2_S.glsl", 3, 1, 4, True, 2, True),
    "Upscale_M": ("Anime4K_Upscale_CNN_x2_M.glsl", 7, 7, 4, False, 2, True),
    "Upscale_L": ("Anime4K_Upscale_CNN_x2_L.glsl", 3, 1, 8, True, 2),
    "Upscale_VL": ("Anime4K_Upscale_CNN_x2_VL.glsl", 7, 7, 8, False, 2),
    "Upscale_UL": ("Anime4K_Upscale_CNN_x2_UL.glsl", 7, 5, 12, False, 2),
    "Upscale_Denoise_S": ("Anime4K_Upscale_Denoise_CNN_x2_S.glsl", 3, 1, 4, True, 2, True),
    "Upscale_Denoise_M": ("Anime4K_Upscale_Denoise_CNN_x2_M.glsl", 7, 7, 4, False, 2, True),
    "Upscale_Denoise_L": ("Anime4K_Upscale_Denoise_CNN_x2_L.glsl", 3, 1, 8, True, 2),
    "Upscale_Denoise_VL": ("Anime4K_Upscale_Denoise_CNN_x2_VL.glsl", 7, 7, 8, False, 2),
    "Upscale_Denoise_UL": ("Anime4K_Upscale_Denoise_CNN_x2_UL.glsl", 7, 5, 12, False, 2),
    "Restore_S": ("Anime4K_Restore_CNN_S.glsl", 3, 1, 4, True, 1),
    #"Restore_M": ("Anime4K_Restore_CNN_M.glsl", 7, 7, 4, False, 1), Doesn't work for some reason
    "Restore_L": ("Anime4K_Restore_CNN_L.glsl", 4, 1, 8, True, 1),
    "Restore_VL": ("Anime4K_Restore_CNN_VL.glsl", 8, 7, 8, False, 1),
    "Restore_UL": ("Anime4K_Restore_CNN_UL.glsl", 8, 5, 12, False, 1),
    "Restore_Soft_S": ("Anime4K_Restore_CNN_Soft_S.glsl", 3, 1, 4, True, 1),
    "Restore_Soft_M": ("Anime4K_Restore_CNN_Soft_M.glsl", 7, 7, 4, False, 1),
    "Restore_Soft_L": ("Anime4K_Restore_CNN_Soft_L.glsl", 4, 1, 8, True, 1),
    "Restore_Soft_VL": ("Anime4K_Restore_CNN_Soft_VL.glsl", 8, 7, 8, False, 1),
    "Restore_Soft_UL": ("Anime4K_Restore_CNN_Soft_UL.glsl", 8, 5, 12, False, 1)
}

def create_model(name, upscale_mode="bilinear"):
    """Instantiate an Anime4K upscaler.

    `name`: see `model_dict` in this module.
    """
    filename, *model_params = model_dict[name]
    filename = os.path.join(os.path.dirname(__file__), "glsl", filename)
    model = anime4k(*model_params, upscale_mode=upscale_mode)
    model.import_param(filename)
    return model

# --------------------------------------------------------------------------------
# Main program

def main():
    #@title Create pipeline and warmup
    USE_FP16 = True  # @param {type:"boolean"}
    OPTIMIZATION = "None"  # @param ["TorchDynamo", "Torch-TensorRT", "OpenVINO", "None"]
    #@markdown Change arguments passing to Anime4KPipeline to implement another preset.
    #@markdown For example, preset A (HQ) is
    #@markdown ```
    #@markdown ~~/shaders/Anime4K_Clamp_Highlights.glsl;~~/shaders/Anime4K_Restore_CNN_VL.glsl;~~/shaders/Anime4K_Upscale_CNN_x2_VL.glsl;~~/shaders/Anime4K_AutoDownscalePre_x2.glsl;~~/shaders/Anime4K_AutoDownscalePre_x4.glsl;~~/shaders/Anime4K_Upscale_CNN_x2_M.glsl
    #@markdown ```
    #@markdown can be implemented as the predefined pipeline. `screen_width` and `screen_height` are size of output images.

    #Implementation of preset A (HQ)
    pipeline = Anime4KPipeline(
        ClampHighlight(),
        create_model("Upscale_Denoise_VL"),
        AutoDownscalePre(4),
        create_model("Upscale_M"),
        # screen_width=1920, screen_height=1080,
        screen_width=512, screen_height=512,
        final_stage_upscale_mode="bilinear"
    )

    def make_rand_image():
        # return torch.rand(1, 3, 720, 1280).to(device)
        return torch.rand(1, 3, 512, 512).to(device)

    if OPTIMIZATION == "TorchDynamo":
        torch._inductor.config.conv_1x1_as_mm = True
        #torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        #torch._inductor.config.coordinate_descent_check_all_directions = True
        torch._inductor.config.use_mixed_mm = True
        device = "cuda"
        if USE_FP16:
            pipeline = pipeline.to(device).half()
        pipeline = torch.compile(pipeline, mode="reduce-overhead")
    elif OPTIMIZATION == "OpenVINO":
        # pip install openvino==2023.3.0
        import openvino.torch  # noqa: F401, I assume this is for side effects
        pipeline = torch.compile(pipeline, backend="openvino")
        device = "cpu"
    elif OPTIMIZATION == "Torch-TensorRT":
        # pip install torch_tensorrt
        import torch_tensorrt  # noqa: F401, I assume this is for side effects
        device = "cuda"
        rand_image = make_rand_image()
        if USE_FP16:
            pipeline = pipeline.to(device).half()
            rand_image = rand_image.half()
        #Is this legal?
        pipeline = torch.compile(torch.jit.script(torch.jit.trace(pipeline, rand_image)), backend="tensorrt")
    else:
        device = "cuda"
        if USE_FP16:
            pipeline = pipeline.to(device).half()

    def timed(fn):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize()
            return result, start.elapsed_time(end) / 1000
        else:
            from time import perf_counter
            start = perf_counter()
            result = fn()
            end = perf_counter()
            return result, end - start
    rand_image = make_rand_image()
    if USE_FP16:
        rand_image = rand_image.half()
    pipeline(rand_image)
    for _ in range(20):
        _, t = timed(lambda: pipeline(rand_image))
        print(t * 1000, 'ms')

    # #@title Export to onnx (Require OPTIMIZATION=None)
    # # pip install --upgrade onnx onnxscript einops
    # # pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    # onnx_filename = "pipeline_preset_C.onnx"  # @param {"type": "string"}
    # torch.onnx.export(pipeline, rand_image, onnx_filename, input_names=["input"])

    # Inference with Onnxruntime + TensorRT
    # Create session, takes about 7min for compiling engine
    # import onnxruntime as ort
    # import numpy as np
    # import locale
    # locale.getpreferredencoding = lambda: "UTF-8"
    # !mkdir /content/trt_cache
    # onnx_filename = "pipeline_preset_A.onnx" #@param {"type": "string"}
    # use_fp16 = True #@param {"type": "boolean"}
    # ort_session = ort.InferenceSession(onnx_filename, providers=[
    #     (
    #         "TensorrtExecutionProvider",
    #         { "trt_fp16_enable": use_fp16 },
    #     ),
    #     "CUDAExecutionProvider",
    #     "CPUExecutionProvider",
    # ])
    #
    # from time import perf_counter
    # inp = np.random.randn(5, 3, 720, 1280).astype(np.float16)
    # for _ in range(20):
    #     start = perf_counter()
    #     outputs = ort_session.run(
    #         None,
    #         {"input": inp},
    #     )
    #     print((perf_counter() - start) * 1000, 'ms')

    # PyTorch Inference

    # wget https://cdni.fancaps.net/file/fancaps-animeimages/6486130.jpg -nc
    image_filename = os.path.join(os.path.dirname(__file__), "images", "6486130.png")
    image = to_tensor(PIL.Image.open(image_filename).convert("RGB")).unsqueeze(0).to(device)
    if USE_FP16:
        image = image.half()
    for _ in range(5):
        out, t = timed(lambda: pipeline(image))
    print(t * 1000, 'ms')
    image = to_pil(out[0])

if __name__ == '__main__':
    main()
