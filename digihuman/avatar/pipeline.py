from __future__ import annotations

import collections
import gc
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import kornia
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn.functional as F
from einops import rearrange
from insightface.app import FaceAnalysis
from torch import Tensor
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from .config import RuntimeSelection

logger = logging.getLogger(__name__)


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]
INSIGHTFACE_DETECT_SIZE = 512


@dataclass
class CachedReferenceFrame:
    face_rgb: np.ndarray
    coords: list[int]
    affine_matrix: np.ndarray
    packed_face: np.ndarray | None = None


def get_video_fps(video_path: str | os.PathLike) -> int:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(fps)


def run_command(command: list[str]) -> None:
    subprocess.run([str(item) for item in command], check=True, capture_output=True, text=True)


def load_fixed_mask(resolution: int, mask_image: np.ndarray) -> Tensor:
    rgb_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_mask, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    return rearrange(torch.from_numpy(resized), "h w c -> c h w")


def get_model_face_size(model_path: str | os.PathLike) -> int:
    name = Path(model_path).stem
    if "_" in name:
        name = name.split("_", 1)[0]
    numbers = re.findall(r"\d+", name)
    if not numbers:
        raise ValueError(f"Unable to infer face size from model path: {model_path}")
    return int(numbers[0])


def resize_tensor_image(image: Tensor, size: tuple[int, int]) -> Tensor:
    tensor = image.unsqueeze(0)
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    resized = F.interpolate(
        tensor,
        size=size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.squeeze(0)


def resize_tensor_batch(images: Tensor, size: tuple[int, int]) -> Tensor:
    tensor = images
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    return F.interpolate(
        tensor,
        size=size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


class AlignRestore:
    def __init__(self, align_points: int = 3, resolution: int = 256, device: str = "cpu", dtype=torch.float32):
        if align_points != 3:
            raise ValueError("Only 3-point alignment is supported")

        ratio = resolution / 256 * 2.8
        self.upscale_factor = 1
        self.crop_ratio = (ratio, ratio)
        self.face_template = np.array([[17, 20], [58, 20], [37.5, 40]]) * ratio
        self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
        self.p_bias = None
        self.device = device
        self.dtype = dtype
        self.fill_value = torch.tensor([127, 127, 127], device=device, dtype=dtype)
        # Dynamic anatomical mask: elliptical soft mask positioned on mouth/chin region.
        # Face template: left_eye=(17,20)*r, right_eye=(58,20)*r, nose=(37.5,40)*r
        # face_size = (75*r, 100*r) → nose is at (50%, 40%) of face height.
        # Mouth is estimated at (50%, ~62%) based on standard face proportions.
        h, w = self.face_size[1], self.face_size[0]

        # Vertical component: smoothstep 0→1 from just below nose (44%) to mouth top (55%)
        blend_y0 = int(h * 0.44)
        blend_y1 = int(h * 0.55)
        vy = np.zeros(h, dtype=np.float32)
        for y in range(h):
            if y >= blend_y1:
                vy[y] = 1.0
            elif y > blend_y0:
                t = (y - blend_y0) / max(blend_y1 - blend_y0, 1)
                vy[y] = t * t * (3.0 - 2.0 * t)  # smoothstep: zero derivative at endpoints

        # Horizontal component: Gaussian centered on face, sigma≈0.26w
        # At ±1σ (~mouth corners) value ≈ 0.61, at ±2σ (cheeks) value ≈ 0.14
        cx = (w - 1) / 2.0
        sigma_x = w * 0.26
        hx = np.exp(-0.5 * ((np.arange(w, dtype=np.float32) - cx) / sigma_x) ** 2)

        mask_np = (vy[:, None] * hx[None, :]).astype(np.float32)

        # Light Gaussian blur for ultra-smooth blend boundary
        ksize = max(int(min(h, w) * 0.08) | 1, 3)
        mask_np = cv2.GaussianBlur(mask_np, (ksize, ksize), 0)
        mask_np = np.clip(mask_np, 0.0, 1.0)

        self.mask = torch.from_numpy(mask_np[None, None]).to(device=device, dtype=dtype)
        self._ellipse_kernel_cache: dict[tuple[int, int, str, str], Tensor] = {}

    def align_warp_face(self, img: np.ndarray, landmarks3: np.ndarray, smooth: bool = True):
        affine_matrix, self.p_bias = self.transformation_from_points(
            landmarks3, self.face_template, smooth, self.p_bias
        )
        img_tensor = rearrange(torch.from_numpy(img).to(device=self.device, dtype=self.dtype), "h w c -> c h w").unsqueeze(0)
        affine_tensor = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        cropped_face = kornia.geometry.transform.warp_affine(
            img_tensor,
            affine_tensor,
            (self.face_size[1], self.face_size[0]),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        )
        cropped_face = rearrange(cropped_face.squeeze(0), "c h w -> h w c").cpu().numpy().astype(np.uint8)
        return cropped_face, affine_matrix

    def _get_ellipse_kernel(self, height: int, width: int) -> Tensor:
        height = max(int(height), 1)
        width = max(int(width), 1)
        cache_key = (height, width, str(self.device), str(self.dtype))
        cached = self._ellipse_kernel_cache.get(cache_key)
        if cached is not None:
            return cached

        y = torch.linspace(-1.0, 1.0, steps=height, device=self.device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, steps=width, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        kernel = ((xx * xx + yy * yy) <= 1.0).to(dtype=self.dtype)
        if torch.count_nonzero(kernel) == 0:
            kernel = torch.ones((height, width), device=self.device, dtype=self.dtype)
        self._ellipse_kernel_cache[cache_key] = kernel
        return kernel

    def restore_batch(
        self,
        input_imgs: np.ndarray | list[np.ndarray],
        faces: Tensor,
        affine_matrices,
        scale_h: float = 1.0,
        scale_w: float = 1.0,
    ) -> np.ndarray:
        input_array = np.asarray(input_imgs)
        if input_array.ndim == 3:
            input_array = np.expand_dims(input_array, axis=0)
        height, width = input_array.shape[1:3]

        if isinstance(affine_matrices, np.ndarray):
            affine_tensor = torch.from_numpy(affine_matrices).to(device=self.device, dtype=self.dtype)
        else:
            affine_tensor = affine_matrices.to(device=self.device, dtype=self.dtype)
        if affine_tensor.ndim == 2:
            affine_tensor = affine_tensor.unsqueeze(0)

        face_tensor = faces.to(device=self.device, dtype=self.dtype)
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)

        inverse_affine = kornia.geometry.transform.invert_affine_transform(affine_tensor)
        inv_face = kornia.geometry.transform.warp_affine(
            face_tensor,
            inverse_affine,
            (height, width),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        ).clamp(0, 1) * 255

        input_tensor = torch.from_numpy(input_array).to(device=self.device, dtype=self.dtype)
        input_tensor = rearrange(input_tensor, "b h w c -> b c h w")

        batch_size = input_tensor.shape[0]
        mask_batch = self.mask.expand(batch_size, -1, -1, -1)
        inv_mask = kornia.geometry.transform.warp_affine(
            mask_batch,
            inverse_affine,
            (height, width),
            padding_mode="zeros",
        )
        base_kernel_size = max(2 * self.upscale_factor, 1)
        base_kernel = torch.ones((base_kernel_size, base_kernel_size), device=self.device, dtype=self.dtype)
        inv_mask_erosion = kornia.morphology.erosion(inv_mask, base_kernel)
        inv_mask_erosion_t = inv_mask_erosion.expand(-1, 3, -1, -1)

        total_face_area = inv_mask_erosion.float().sum(dim=(1, 2, 3))
        w_edge = torch.clamp((total_face_area.sqrt() / 20.0).to(torch.int64), min=1)
        erosion_radius = torch.clamp(w_edge * 2, min=1)
        kernel_h = max(int(torch.max(torch.ceil(erosion_radius.float() * float(scale_h))).item()), 1)
        kernel_w = max(int(torch.max(torch.ceil(erosion_radius.float() * float(scale_w))).item()), 1)
        # 大 kernel 的 kornia erosion 内部用 F.unfold，4K 帧 + 大 kernel 会 OOM（TB 级显存需求）
        # 改为 cv2.erode（CPU），对每张 mask 单独处理后搬回 GPU
        ellipse_cv2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
        inv_mask_center_np = inv_mask_erosion.squeeze(1).cpu().numpy()  # (B, H, W)
        eroded_list = [cv2.erode((m * 255).astype(np.uint8), ellipse_cv2).astype(np.float32) / 255.0 for m in inv_mask_center_np]
        inv_mask_center = torch.from_numpy(np.stack(eroded_list, axis=0)).unsqueeze(1).to(device=self.device, dtype=self.dtype)

        mask_for_color = (inv_mask_erosion_t > 0.5).to(dtype=self.dtype)
        mask_counts = mask_for_color.sum(dim=(2, 3)).clamp(min=1.0)
        valid_channels = mask_counts > 100.0
        gen_mean = (inv_face * mask_for_color).sum(dim=(2, 3)) / mask_counts
        orig_mean = (input_tensor * mask_for_color).sum(dim=(2, 3)) / mask_counts
        gen_var = (((inv_face - gen_mean[..., None, None]) ** 2) * mask_for_color).sum(dim=(2, 3)) / mask_counts
        orig_var = (((input_tensor - orig_mean[..., None, None]) ** 2) * mask_for_color).sum(dim=(2, 3)) / mask_counts
        gen_std = gen_var.sqrt().clamp(min=1.0)
        orig_std = orig_var.sqrt().clamp(min=1.0)
        adjusted_face = (
            (inv_face - gen_mean[..., None, None]) * (orig_std / gen_std)[..., None, None]
            + orig_mean[..., None, None]
        )
        inv_face = torch.where(valid_channels[..., None, None], adjusted_face, inv_face).clamp(0, 255)

        pasted_face = inv_mask_erosion_t * inv_face

        blur_size = max(int(torch.max(w_edge * 4 + 1).item()), 3)
        if blur_size % 2 == 0:
            blur_size += 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        inv_soft_mask = kornia.filters.gaussian_blur2d(
            inv_mask_center,
            (blur_size, blur_size),
            (sigma, sigma),
        )
        inv_soft_mask_3d = inv_soft_mask.expand(-1, 3, -1, -1)
        blended = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_tensor
        blended = rearrange(blended, "b c h w -> b h w c").contiguous().clamp(0, 255).to(dtype=torch.uint8)
        return blended.cpu().numpy()

    def restore_img(self, input_img: np.ndarray, face: Tensor, affine_matrix, scale_h: float = 1.0, scale_w: float = 1.0):
        return self.restore_batch(
            input_imgs=np.expand_dims(input_img, axis=0),
            faces=face.unsqueeze(0),
            affine_matrices=np.expand_dims(affine_matrix, axis=0),
            scale_h=scale_h,
            scale_w=scale_w,
        )[0]

    def transformation_from_points(self, points1, points0, smooth: bool = True, p_bias=None):
        points2 = torch.tensor(points0, device=self.device, dtype=torch.float32) if isinstance(points0, np.ndarray) else points0.clone()
        points1_tensor = torch.tensor(points1, device=self.device, dtype=torch.float32) if isinstance(points1, np.ndarray) else points1.clone()

        c1 = torch.mean(points1_tensor, dim=0)
        c2 = torch.mean(points2, dim=0)
        points1_centered = points1_tensor - c1
        points2_centered = points2 - c2
        s1 = torch.std(points1_centered)
        s2 = torch.std(points2_centered)
        points1_normalized = points1_centered / s1
        points2_normalized = points2_centered / s2

        covariance = torch.matmul(points1_normalized.T, points2_normalized)
        u, _, v = torch.svd(covariance)
        rotation = torch.matmul(v, u.T)
        if torch.det(rotation) < 0:
            v[:, -1] = -v[:, -1]
            rotation = torch.matmul(v, u.T)

        scaled_rotation = (s2 / s1) * rotation
        translation = c2.reshape(2, 1) - (s2 / s1) * torch.matmul(rotation, c1.reshape(2, 1))
        matrix = torch.cat((scaled_rotation, translation), dim=1)

        if smooth:
            bias = points2_normalized[2] - points1_normalized[2]
            bias = bias if p_bias is None else p_bias * 0.2 + bias * 0.8
            p_bias = bias
            matrix[:, 2] = matrix[:, 2] + bias

        return matrix.cpu().numpy(), p_bias


class FaceDetector:
    def __init__(self, auxiliary_path: str | os.PathLike, runtime: RuntimeSelection):
        providers = list(runtime.onnx_providers)
        ctx_id = 0 if runtime.resolved == "cuda" else -1
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            root=str(auxiliary_path),
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))

    def __call__(self, frame: np.ndarray, threshold: float = 0.5):
        frame_height, frame_width, _ = frame.shape
        faces = self.app.get(frame)
        selected_face = None
        max_size = 0

        for face in faces:
            bbox = face.bbox.astype(np.int_).tolist()
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width < 50 or height < 80:
                continue
            if width / height > 1.5 or width / height < 0.2:
                continue
            if face.det_score < threshold:
                continue
            size = width * height
            if size > max_size:
                max_size = size
                selected_face = face

        if selected_face is None:
            return None, None

        landmarks = np.round(selected_face.landmark_2d_106).astype(np.int_)
        middle_face = np.mean([landmarks[74], landmarks[73]], axis=0)
        sub_landmarks = landmarks[LMK_ADAPT_ORIGIN_ORDER]
        middle_distance = np.max(sub_landmarks[:, 1]) - middle_face[1]
        upper_bound = middle_face[1] - middle_distance
        x1, y1, x2, y2 = (
            np.min(sub_landmarks[:, 0]),
            int(upper_bound),
            np.max(sub_landmarks[:, 0]),
            np.max(sub_landmarks[:, 1]),
        )

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            x1, y1, x2, y2 = selected_face.bbox.astype(np.int_).tolist()

        y2 += int((x2 - x1) * 0.1)
        x1 -= int((x2 - x1) * 0.05)
        x2 += int((x2 - x1) * 0.05)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)
        return (x1, y1, x2, y2), landmarks


class ImageProcessor:
    def __init__(
        self,
        resolution: int,
        device: str,
        runtime: RuntimeSelection,
        auxiliary_path: str | os.PathLike,
        mask_image: np.ndarray,
        dtype=torch.float32,
    ):
        self.resolution = resolution
        self.resize = lambda image: resize_tensor_image(image, (resolution, resolution))
        self.restorer = AlignRestore(resolution=resolution, device=device, dtype=dtype)
        self.mask_image = load_fixed_mask(resolution, mask_image)
        self.face_detector = FaceDetector(auxiliary_path=auxiliary_path, runtime=runtime)

    def affine_transform(self, image: np.ndarray):
        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            raise RuntimeError("Face not detected")

        left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)
        right_eye = np.mean(landmark_2d_106[101:106], axis=0)
        nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)
        landmarks3 = np.round([left_eye, right_eye, nose])

        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: Tensor):
        image = self.resize(image)
        pixel_values = image / 255.0
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[Tensor, np.ndarray]):
        images_tensor = torch.from_numpy(images) if isinstance(images, np.ndarray) else images
        if images_tensor.shape[3] == 3:
            images_tensor = rearrange(images_tensor, "f h w c -> f c h w")
        results = [self.preprocess_fixed_mask_image(image) for image in images_tensor]
        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)


class LstmSync:
    def __init__(
        self,
        human_path: str | os.PathLike,
        hubert_path: str | os.PathLike,
        checkpoints_dir: str | os.PathLike,
        runtime: RuntimeSelection,
        batch_size: int = 4,
        sync_offset: int = 0,
        scale_h: float = 1.0,
        scale_w: float = 1.0,
        compress_inference_check_box: bool = False,
        ffmpeg_bin: str = "ffmpeg",
        progress_callback=None,
    ):
        self.human_path = Path(human_path).expanduser().resolve()
        self.hubert_path = Path(hubert_path).expanduser().resolve()
        self.checkpoints_dir = Path(checkpoints_dir).expanduser().resolve()
        self.runtime = runtime
        self.face_size = get_model_face_size(self.human_path)
        self.wav2lip_batch_size = batch_size
        self.syncnet_T = 64  # 扩大LSTM状态窗口，减少帧间断裂感
        self.audio_type = "hubert"
        self.sync_offset = sync_offset
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.compress_inference_check_box = compress_inference_check_box
        self.model_dtype = np.float16 if runtime.resolved == "cuda" else np.float32
        self.device = runtime.torch_device
        self.ffmpeg_bin = ffmpeg_bin
        self._progress_callback = progress_callback

        # Optimization: Cache ONNX session with optimized options
        self._onnx_session: ort.InferenceSession | None = None
        # Optimization: Cache HuBERT model and feature extractor
        self._hubert_model: HubertModel | None = None
        self._feature_extractor: Wav2Vec2FeatureExtractor | None = None
        # Cache face detection/alignment results for repeated avatar requests.
        self._reference_cache: collections.OrderedDict[str, list[CachedReferenceFrame]] = collections.OrderedDict()
        self._reference_cache_max_entries = max(int(os.getenv("DIGIHUMAN_REFERENCE_CACHE_SIZE", "2")), 0)
        self._reference_prepack_enabled = os.getenv("DIGIHUMAN_REFERENCE_PREPACK", "1").lower() not in {"0", "false", "no"}
        self.last_run_stats: dict[str, float] = {}

        repair_npy_path = self.checkpoints_dir / "repair.npy"
        auxiliary_path = self.checkpoints_dir / "auxiliary"
        if not repair_npy_path.exists():
            raise FileNotFoundError(f"repair.npy not found: {repair_npy_path}")
        if not auxiliary_path.exists():
            raise FileNotFoundError(f"auxiliary model directory not found: {auxiliary_path}")

        mask_image = np.load(repair_npy_path)
        self.detect_face = ImageProcessor(
            resolution=self.face_size,
            device=self.device,
            runtime=runtime,
            auxiliary_path=auxiliary_path,
            mask_image=mask_image,
            dtype=torch.float32,
        )

    def configure_request(
        self,
        batch_size: int,
        sync_offset: int,
        scale_h: float,
        scale_w: float,
        compress_inference_check_box: bool,
        progress_callback=None,
    ) -> None:
        self.wav2lip_batch_size = batch_size
        self.sync_offset = sync_offset
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.compress_inference_check_box = compress_inference_check_box
        self._progress_callback = progress_callback

    def preload_models(self) -> None:
        self._get_onnx_session()
        self._get_hubert()

    @staticmethod
    def _torch_dtype_from_numpy(np_dtype) -> torch.dtype:
        return torch.float16 if np_dtype == np.float16 else torch.float32

    @staticmethod
    def _numpy_dtype_from_ort(type_name: str):
        return np.float16 if "float16" in type_name else np.float32

    def _bind_tensor_input(self, io_binding, name: str, tensor: Tensor, np_dtype) -> None:
        io_binding.bind_input(name, "cuda", 0, np_dtype, tuple(tensor.shape), tensor.data_ptr())

    def _bind_tensor_output(self, io_binding, name: str, tensor: Tensor, np_dtype) -> None:
        io_binding.bind_output(name, "cuda", 0, np_dtype, tuple(tensor.shape), tensor.data_ptr())

    def _run_gpu_recurrent_inference(
        self,
        session: ort.InferenceSession,
        input_names: list[str],
        output_names: list[str],
        img_batch: np.ndarray,
        mel_batch: np.ndarray,
        sequence_offset: int,
        hn_state: Tensor | None,
        cn_state: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        torch_dtype = self._torch_dtype_from_numpy(self.model_dtype)
        img_batch_tensor = torch.from_numpy(np.ascontiguousarray(img_batch)).to(device=self.device, dtype=torch_dtype)
        mel_batch_tensor = torch.from_numpy(np.ascontiguousarray(mel_batch)).to(device=self.device, dtype=torch_dtype)

        x_frame_buffer = torch.empty_like(img_batch_tensor[0:1])
        indiv_frame_buffer = torch.empty_like(mel_batch_tensor[0:1])
        hn_in = (
            hn_state.to(device=self.device, dtype=torch_dtype)
            if hn_state is not None
            else torch.zeros((2, 1, 576), device=self.device, dtype=torch_dtype)
        )
        cn_in = (
            cn_state.to(device=self.device, dtype=torch_dtype)
            if cn_state is not None
            else torch.zeros((2, 1, 576), device=self.device, dtype=torch_dtype)
        )
        generated_buffer = torch.empty(
            (1, 3, self.face_size, self.face_size),
            device=self.device,
            dtype=torch_dtype,
        )
        hn_out = torch.empty_like(hn_in)
        cn_out = torch.empty_like(cn_in)
        predictions = torch.empty(
            (mel_batch_tensor.shape[0], 3, self.face_size, self.face_size),
            device=self.device,
            dtype=torch_dtype,
        )

        io_binding = session.io_binding()
        input_np_dtype = self.model_dtype
        output_np_dtypes = [self._numpy_dtype_from_ort(output.type) for output in session.get_outputs()]

        self._bind_tensor_input(io_binding, input_names[0], indiv_frame_buffer, input_np_dtype)
        self._bind_tensor_input(io_binding, input_names[1], x_frame_buffer, input_np_dtype)
        self._bind_tensor_input(io_binding, input_names[2], hn_in, input_np_dtype)
        self._bind_tensor_input(io_binding, input_names[3], cn_in, input_np_dtype)
        self._bind_tensor_output(io_binding, output_names[0], generated_buffer, output_np_dtypes[0])
        self._bind_tensor_output(io_binding, output_names[1], hn_out, output_np_dtypes[1])
        self._bind_tensor_output(io_binding, output_names[2], cn_out, output_np_dtypes[2])

        for frame_index in range(mel_batch_tensor.shape[0]):
            global_frame_index = sequence_offset + frame_index
            if global_frame_index == 0 or ((global_frame_index + 1) % self.syncnet_T == 0):
                hn_in.zero_()
                cn_in.zero_()

            x_frame_buffer.copy_(img_batch_tensor[frame_index : frame_index + 1], non_blocking=True)
            indiv_frame_buffer.copy_(mel_batch_tensor[frame_index : frame_index + 1], non_blocking=True)

            if hasattr(io_binding, "synchronize_inputs"):
                io_binding.synchronize_inputs()
            session.run_with_iobinding(io_binding)
            if hasattr(io_binding, "synchronize_outputs"):
                io_binding.synchronize_outputs()

            predictions[frame_index].copy_(generated_buffer[0])
            hn_in.copy_(hn_out)
            cn_in.copy_(cn_out)

        return predictions, hn_in, cn_in

    def _report(self, step: str, progress: int, message: str) -> None:
        if self._progress_callback:
            self._progress_callback(step, progress, message)

    def _get_onnx_session(self) -> ort.InferenceSession:
        if self._onnx_session is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 8  # L20: 8 vCPU
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True

            providers = list(self.runtime.onnx_providers)
            provider_options = []
            for p in providers:
                if p == "TensorrtExecutionProvider":
                    trt_cache = Path(self.checkpoints_dir) / "trt_cache"
                    trt_cache.mkdir(exist_ok=True)
                    provider_options.append({
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(trt_cache),
                        "trt_fp16_enable": True,       # L20 FP16 加速
                        "trt_max_workspace_size": str(4 * 1024 * 1024 * 1024),  # 4GB workspace
                    })
                elif p == "CUDAExecutionProvider":
                    provider_options.append({
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    })
                else:
                    provider_options.append({})

            self._onnx_session = ort.InferenceSession(
                str(self.human_path), sess_options,
                providers=providers,
                provider_options=provider_options,
            )
            applied = self._onnx_session.get_providers()
            logger.info("ONNX session created | providers: %s", applied)
        return self._onnx_session

    def _get_hubert(self) -> tuple[Wav2Vec2FeatureExtractor, HubertModel]:
        if self._hubert_model is None or self._feature_extractor is None:
            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(self.hubert_path))
            self._hubert_model = HubertModel.from_pretrained(str(self.hubert_path)).to(self.device).eval()
            if self.runtime.resolved == "cuda":
                self._hubert_model = self._hubert_model.half()
            logger.info("HuBERT model loaded and cached")
        return self._feature_extractor, self._hubert_model

    def _face_detect(self, images: list[np.ndarray]):
        results: list[CachedReferenceFrame] = []
        for image in tqdm(images, desc="Detecting face"):
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face, box, affine_matrix = self.detect_face.affine_transform(frame)
            face = rearrange(face.cpu().numpy(), "c h w -> h w c")
            packed_face = self._prepare_packed_face(face) if self._reference_prepack_enabled else None
            results.append(
                CachedReferenceFrame(
                    face_rgb=face,
                    coords=box,
                    affine_matrix=affine_matrix,
                    packed_face=packed_face,
                )
            )
        return results

    def _prepare_packed_face(self, face_rgb: np.ndarray) -> np.ndarray:
        ref_pixel_values, masked_pixel_values, masks = self.detect_face.prepare_masks_and_masked_images(
            np.expand_dims(face_rgb, axis=0)
        )
        packed_face = np.concatenate((masks, masked_pixel_values, ref_pixel_values), axis=1)[0]
        return np.asarray(packed_face, dtype=self.model_dtype)

    def _datagen(
        self,
        frames: list[np.ndarray],
        reps: list[np.ndarray],
        face_det_results: list[CachedReferenceFrame],
    ):
        img_batch = []
        mel_batch = []
        frame_batch = []
        coords_batch = []
        affines_batch = []

        for index, rep in enumerate(reps):
            frame_count = len(frames)
            if index // frame_count % 2 == 0:
                frame_index = index % frame_count
            else:
                frame_index = frame_count - 1 - index % frame_count

            frame_to_save = frames[frame_index]
            cached_frame = face_det_results[frame_index]
            packed_face = cached_frame.packed_face
            if packed_face is None:
                packed_face = self._prepare_packed_face(cached_frame.face_rgb)
                cached_frame.packed_face = packed_face
            img_batch.append(packed_face)
            mel_batch.append(rep)
            frame_batch.append(frame_to_save)
            coords_batch.append(cached_frame.coords)
            affines_batch.append(cached_frame.affine_matrix)

            if len(img_batch) >= self.wav2lip_batch_size:
                yield self._pack_batch(img_batch, mel_batch, frame_batch, coords_batch, affines_batch)
                img_batch, mel_batch, frame_batch, coords_batch, affines_batch = [], [], [], [], []

        if img_batch:
            yield self._pack_batch(img_batch, mel_batch, frame_batch, coords_batch, affines_batch)

    def _pack_batch(
        self,
        img_batch: list[np.ndarray],
        mel_batch: list[np.ndarray],
        frame_batch: list[np.ndarray],
        coords_batch: list[list[int]],
        affines_batch: list[np.ndarray],
    ):
        img_array = np.asarray(img_batch, dtype=self.model_dtype)
        mel_array = np.asarray(mel_batch)
        return img_array, mel_array, frame_batch, coords_batch, affines_batch

    def _get_reference_cache_key(self, video_path: Path) -> str:
        stat = video_path.stat()
        return f"{video_path}:{stat.st_mtime_ns}:{stat.st_size}:{self.face_size}"

    def _get_cached_face_results(self, video_path: Path):
        if self._reference_cache_max_entries <= 0:
            return None
        cache_key = self._get_reference_cache_key(video_path)
        cached = self._reference_cache.get(cache_key)
        if cached is None:
            return None
        self._reference_cache.move_to_end(cache_key)
        logger.info("Reference cache hit: %s", video_path.name)
        return cached

    def _store_cached_face_results(self, video_path: Path, face_det_results: list[CachedReferenceFrame]) -> None:
        if self._reference_cache_max_entries <= 0:
            return
        cache_key = self._get_reference_cache_key(video_path)
        self._reference_cache[cache_key] = face_det_results
        self._reference_cache.move_to_end(cache_key)
        while len(self._reference_cache) > self._reference_cache_max_entries:
            evicted_key, _ = self._reference_cache.popitem(last=False)
            logger.info("Reference cache evicted: %s", evicted_key)

    def run(
        self,
        video_path: str | os.PathLike,
        video_fps25_path: str | os.PathLike,
        video_temp_path: str | os.PathLike,
        audio_path: str | os.PathLike,
        audio_temp_path: str | os.PathLike,
        video_out_path: str | os.PathLike,
        compress_inference_check_box: bool | None = None,
    ) -> Path:
        if compress_inference_check_box is not None:
            self.compress_inference_check_box = compress_inference_check_box
        stats: dict[str, float] = {}
        total_start = time.perf_counter()

        self._report("preprocessing", 5, "视频预处理中...")
        source_video_path = Path(video_path).expanduser().resolve()
        fps25_path = Path(video_fps25_path).expanduser().resolve()
        source_audio_path = Path(audio_path).expanduser().resolve()
        temp_audio_path = Path(audio_temp_path).expanduser().resolve()
        final_video_path = Path(video_out_path).expanduser().resolve()
        reference_cache_source = source_video_path

        session = self._get_onnx_session()
        input_names = [input_info.name for input_info in session.get_inputs()]
        output_names = [output_info.name for output_info in session.get_outputs()]
        model_input_type = session.get_inputs()[0].type
        self.model_dtype = np.float16 if "float16" in model_input_type else np.float32

        normalize_start = time.perf_counter()
        if get_video_fps(source_video_path) != 25:
            if not self.compress_inference_check_box:
                if not self._try_nvenc_fps_normalize(source_video_path, fps25_path):
                    run_command(
                        [
                            self.ffmpeg_bin,
                            "-y",
                            "-i",
                            str(source_video_path),
                            "-r",
                            "25",
                            "-c:v",
                            "libx264",
                            "-crf",
                            "16",
                            "-preset",
                            "superfast",
                            "-tune",
                            "film",
                            "-pix_fmt",
                            "yuv420p",
                            "-c:a",
                            "copy",
                            str(fps25_path),
                        ]
                    )
            else:
                run_command([self.ffmpeg_bin, "-y", "-i", str(source_video_path), "-r", "25", str(fps25_path)])
            normalized_video = fps25_path
        else:
            normalized_video = source_video_path
        stats["normalize_video_seconds"] = round(time.perf_counter() - normalize_start, 4)

        read_video_start = time.perf_counter()
        video_stream = cv2.VideoCapture(str(normalized_video))
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
            full_frames.append(frame)
        video_stream.release()

        if not full_frames:
            raise RuntimeError(f"No frames could be read from reference video: {normalized_video}")
        stats["read_frames_seconds"] = round(time.perf_counter() - read_video_start, 4)
        stats["reference_frame_count"] = float(len(full_frames))

        audio_prep_start = time.perf_counter()
        run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-i",
                str(source_audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(temp_audio_path),
            ]
        )
        stats["prepare_audio_seconds"] = round(time.perf_counter() - audio_prep_start, 4)

        self._report("audio_features", 20, "提取音频特征...")
        feature_start = time.perf_counter()
        feature_extractor, hubert_model = self._get_hubert()
        wav, sample_rate = sf.read(str(temp_audio_path))
        input_values = feature_extractor(wav, sampling_rate=sample_rate, return_tensors="pt").input_values
        input_values = input_values.to(self.device)
        if self.runtime.resolved == "cuda":
            input_values = input_values.half()

        with torch.no_grad():
            outputs = hubert_model(input_values)
            reps = outputs.last_hidden_state.permute(0, 2, 1).cpu().numpy()
        stats["audio_features_seconds"] = round(time.perf_counter() - feature_start, 4)

        rep_step_size = 10
        rep_chunks = []
        rep_idx_multiplier = 50.0 / fps
        index = 0
        while True:
            start_idx = int(max(index + self.sync_offset, 0) * rep_idx_multiplier)
            if start_idx + rep_step_size > reps.shape[-1]:
                rep_chunks.append(reps[0, :, reps.shape[-1] - rep_step_size :])
                break
            rep_chunks.append(reps[0, :, start_idx : start_idx + rep_step_size])
            index += 1
        stats["audio_chunk_count"] = float(len(rep_chunks))

        self._report("face_detection", 35, "人脸检测...")
        frame_height, frame_width = full_frames[0].shape[:-1]
        total_batches = int(np.ceil(float(len(full_frames)) / self.wav2lip_batch_size))
        face_prep_start = time.perf_counter()
        face_det_results = self._get_cached_face_results(reference_cache_source)
        cache_hit = face_det_results is not None
        if face_det_results is None:
            face_det_results = self._face_detect(full_frames)
            self._store_cached_face_results(reference_cache_source, face_det_results)
        else:
            self._report("face_detection", 35, "人脸检测已命中缓存...")
        stats["reference_cache_hit"] = 1.0 if cache_hit else 0.0
        stats["prepare_reference_seconds"] = round(time.perf_counter() - face_prep_start, 4)

        # 时序平滑缓冲：用最近N帧的加权平均消除嘴部抖动
        _SMOOTH_WINDOW = 3
        _smooth_weights = np.array([0.25, 0.35, 0.40], dtype=np.float32)  # 越新权重越大
        smooth_buf: collections.deque = collections.deque(maxlen=_SMOOTH_WINDOW)
        sequence_offset = 0
        hn = None
        cn = None

        # Optimization: Use FFmpeg pipe instead of cv2.VideoWriter to encode directly to H.264
        use_nvenc = self.runtime.resolved == "cuda" and self._check_nvenc_available()
        ffmpeg_cmd = self._build_ffmpeg_pipe_cmd(frame_width, frame_height, int(fps), temp_audio_path, final_video_path, use_nvenc=use_nvenc)
        logger.info("Starting FFmpeg pipe: %s", " ".join(ffmpeg_cmd))
        # stdout=DEVNULL 避免 FFmpeg stdout 缓冲区满导致 pipe deadlock
        pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        inference_seconds = 0.0

        try:
            for batch_index, (img_batch, mel_batch, frames, coords, affines) in enumerate(
                tqdm(
                    self._datagen(full_frames, rep_chunks, face_det_results),
                    total=total_batches,
                    desc="Generating video",
                )
            ):
                pct = 40 + int(50 * batch_index / max(total_batches, 1))
                self._report("inference", pct, f"推理中 {batch_index * self.wav2lip_batch_size}/{len(rep_chunks)}")
                batch_start = time.perf_counter()
                mel_batch = np.transpose(mel_batch, (0, 2, 1))
                if self.runtime.resolved == "cuda":
                    prediction_tensor, hn, cn = self._run_gpu_recurrent_inference(
                        session=session,
                        input_names=input_names,
                        output_names=output_names,
                        img_batch=np.ascontiguousarray(img_batch.astype(self.model_dtype)),
                        mel_batch=np.ascontiguousarray(mel_batch.astype(self.model_dtype)),
                        sequence_offset=sequence_offset,
                        hn_state=hn,
                        cn_state=cn,
                    )
                    prediction_tensor = prediction_tensor[:, [2, 1, 0], :, :]
                    target_h = int(coords[0][3] - coords[0][1])
                    target_w = int(coords[0][2] - coords[0][0])
                    resized_batch = resize_tensor_batch(prediction_tensor, size=(target_h, target_w))
                else:
                    generated_frames = []
                    for frame_index in range(mel_batch.shape[0]):
                        global_frame_index = sequence_offset + frame_index
                        if global_frame_index == 0 or ((global_frame_index + 1) % self.syncnet_T == 0):
                            hn = np.zeros((2, 1, 576), dtype=self.model_dtype)
                            cn = np.zeros((2, 1, 576), dtype=self.model_dtype)

                        x_frame = np.expand_dims(img_batch[frame_index, :, :, :].astype(self.model_dtype), axis=0)
                        indiv_frame = np.expand_dims(mel_batch[frame_index, :, :].astype(self.model_dtype), axis=0)
                        generated, hn, cn = session.run(
                            None,
                            {
                                input_names[0]: indiv_frame,
                                input_names[1]: x_frame,
                                input_names[2]: hn,
                                input_names[3]: cn,
                            },
                        )
                        generated_frames.append(generated.squeeze().astype(np.float32))

                    prediction = np.stack(generated_frames, axis=0)[:, [2, 1, 0], :, :]
                    target_h = int(coords[0][3] - coords[0][1])
                    target_w = int(coords[0][2] - coords[0][0])
                    pred_batch = torch.from_numpy(prediction)
                    resized_batch = resize_tensor_batch(pred_batch, size=(target_h, target_w))

                smoothed_frames = []
                for resized in resized_batch:
                    smooth_buf.append(resized.float())
                    n = len(smooth_buf)
                    if n == 1:
                        smoothed = smooth_buf[0]
                    else:
                        w = torch.from_numpy(_smooth_weights[-n:]).to(
                            device=smooth_buf[0].device, dtype=smooth_buf[0].dtype
                        )
                        w = w / w.sum()
                        smoothed = sum(w[i] * smooth_buf[i] for i in range(n))
                    smoothed_frames.append(smoothed)

                merged_batch = self.detect_face.restorer.restore_batch(
                    input_imgs=np.stack(frames, axis=0),
                    faces=torch.stack(smoothed_frames, dim=0),
                    affine_matrices=np.stack(affines, axis=0),
                    scale_h=self.scale_h,
                    scale_w=self.scale_w,
                )
                for merged in merged_batch:
                    pipe.stdin.write(merged.tobytes())
                sequence_offset += mel_batch.shape[0]
                inference_seconds += time.perf_counter() - batch_start

                if (batch_index + 1) % 50 == 0:
                    gc.collect()
                    if self.runtime.resolved == "cuda":
                        torch.cuda.empty_cache()
        finally:
            if pipe.stdin:
                pipe.stdin.close()
                pipe.stdin = None
            self._report("compositing", 92, "等待编码完成...")
            compose_start = time.perf_counter()
            _, stderr_bytes = pipe.communicate()
            stderr_output = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
            if pipe.returncode != 0:
                logger.error("FFmpeg pipe failed (rc=%d): %s", pipe.returncode, stderr_output)
                raise RuntimeError(f"FFmpeg encoding failed: {stderr_output[-500:]}")
            gc.collect()
            if self.runtime.resolved == "cuda":
                torch.cuda.empty_cache()
            stats["compositing_seconds"] = round(time.perf_counter() - compose_start, 4)

        stats["inference_loop_seconds"] = round(inference_seconds, 4)
        stats["total_seconds"] = round(time.perf_counter() - total_start, 4)
        self.last_run_stats = stats
        logger.info("Digital human stats: %s", stats)
        self._report("completed", 100, "完成")
        return final_video_path

    def _build_ffmpeg_pipe_cmd(
        self, width: int, height: int, fps: int, audio_path: Path, output_path: Path,
        use_nvenc: bool = False,
    ) -> list[str]:
        """Build FFmpeg command for piped raw video input with audio muxing."""
        cmd = [
            self.ffmpeg_bin, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-i", str(audio_path),
        ]
        if use_nvenc:
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "18"])
        else:
            cmd.extend(["-c:v", "libx264", "-crf", "16", "-preset", "superfast"])
        cmd.extend([
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-shortest", str(output_path),
        ])
        return cmd

    def _check_nvenc_available(self) -> bool:
        """Check if h264_nvenc encoder is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_bin, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5,
            )
            return "h264_nvenc" in result.stdout
        except Exception:
            return False

    def _try_nvenc_fps_normalize(self, source_path: Path, dest_path: Path) -> bool:
        try:
            run_command(
                [
                    self.ffmpeg_bin,
                    "-y",
                    "-i",
                    str(source_path),
                    "-r",
                    "25",
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p4",
                    "-rc",
                    "vbr",
                    "-cq",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "copy",
                    str(dest_path),
                ]
            )
            return True
        except Exception:
            logger.info("NVENC not available for FPS normalization, falling back to libx264")
            return False
