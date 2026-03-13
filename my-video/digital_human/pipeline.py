from __future__ import annotations

import gc
import logging
import os
import re
import subprocess
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
        self.mask = torch.ones((1, 1, self.face_size[1], self.face_size[0]), device=device, dtype=dtype)

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

    def restore_img(self, input_img: np.ndarray, face: Tensor, affine_matrix, scale_h: float = 1.0, scale_w: float = 1.0):
        height, width, _ = input_img.shape
        if isinstance(affine_matrix, np.ndarray):
            affine_tensor = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        else:
            affine_tensor = affine_matrix.to(device=self.device, dtype=self.dtype)

        inverse_affine = kornia.geometry.transform.invert_affine_transform(affine_tensor)
        face_tensor = face.to(dtype=self.dtype).unsqueeze(0)
        inv_face = kornia.geometry.transform.warp_affine(
            face_tensor,
            inverse_affine,
            (height, width),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        ).squeeze(0)
        inv_face = inv_face.clamp(0, 1) * 255

        input_tensor = rearrange(torch.from_numpy(input_img).to(device=self.device, dtype=self.dtype), "h w c -> c h w")
        inv_mask = kornia.geometry.transform.warp_affine(self.mask, inverse_affine, (height, width), padding_mode="zeros")
        inv_mask_erosion = kornia.morphology.erosion(
            inv_mask,
            torch.ones((2 * self.upscale_factor, 2 * self.upscale_factor), device=self.device, dtype=self.dtype),
        )

        inv_mask_erosion_t = inv_mask_erosion.squeeze(0).expand_as(inv_face)
        pasted_face = inv_mask_erosion_t * inv_face
        total_face_area = torch.sum(inv_mask_erosion.float())
        w_edge = int(total_face_area ** 0.5) // 20
        erosion_radius = w_edge * 2

        inv_mask_erosion_cpu = inv_mask_erosion.squeeze().cpu().numpy().astype(np.float32)
        inv_mask_center = cv2.erode(
            inv_mask_erosion_cpu,
            np.ones((max(int(erosion_radius * scale_h), 1), max(int(erosion_radius * scale_w), 1)), np.uint8),
        )
        inv_mask_center_tensor = torch.from_numpy(inv_mask_center).to(device=self.device, dtype=self.dtype)[None, None, ...]

        blur_size = w_edge * 2 + 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        inv_soft_mask = kornia.filters.gaussian_blur2d(
            inv_mask_center_tensor, (blur_size, blur_size), (sigma, sigma)
        ).squeeze(0)
        inv_soft_mask_3d = inv_soft_mask.expand_as(inv_face)
        blended = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_tensor
        blended = rearrange(blended, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        return blended.cpu().numpy()

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
        self.syncnet_T = 16
        self.audio_type = "hubert"
        self.sync_offset = sync_offset
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.compress_inference_check_box = compress_inference_check_box
        self.model_dtype = np.float16 if runtime.resolved == "cuda" else np.float32
        self.device = runtime.torch_device
        self.ffmpeg_bin = ffmpeg_bin
        self._progress_callback = progress_callback

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

    def _report(self, step: str, progress: int, message: str) -> None:
        if self._progress_callback:
            self._progress_callback(step, progress, message)

    def _face_detect(self, images: list[np.ndarray]):
        faces = []
        boxes = []
        affine_matrices = []
        for image in tqdm(images, desc="Detecting face"):
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face, box, affine_matrix = self.detect_face.affine_transform(frame)
            face = rearrange(face.cpu().numpy(), "c h w -> h w c")
            face = face[..., ::-1]
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)
        return [[face, box, affine] for face, box, affine in zip(faces, boxes, affine_matrices)]

    def _datagen(self, frames: list[np.ndarray], reps: list[np.ndarray]):
        img_batch = []
        mel_batch = []
        frame_batch = []
        coords_batch = []
        affines_batch = []
        face_det_results = self._face_detect(frames)

        for index, rep in enumerate(reps):
            frame_count = len(frames)
            if index // frame_count % 2 == 0:
                frame_index = index % frame_count
            else:
                frame_index = frame_count - 1 - index % frame_count

            frame_to_save = frames[frame_index].copy()
            face, coords, affine_matrix = face_det_results[frame_index].copy()
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            img_batch.append(face)
            mel_batch.append(rep)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)
            affines_batch.append(affine_matrix)

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
        img_array = np.asarray(img_batch)
        mel_array = np.asarray(mel_batch)
        ref_pixel_values, masked_pixel_values, masks = self.detect_face.prepare_masks_and_masked_images(img_array)
        packed_img = np.concatenate((masks, masked_pixel_values, ref_pixel_values), axis=1)
        return packed_img, mel_array, frame_batch, coords_batch, affines_batch

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

        self._report("preprocessing", 5, "视频预处理中...")
        source_video_path = Path(video_path).expanduser().resolve()
        fps25_path = Path(video_fps25_path).expanduser().resolve()
        temp_video_path = Path(f"{video_temp_path}.mp4").expanduser().resolve()
        source_audio_path = Path(audio_path).expanduser().resolve()
        temp_audio_path = Path(audio_temp_path).expanduser().resolve()
        final_video_path = Path(video_out_path).expanduser().resolve()

        session = ort.InferenceSession(str(self.human_path), providers=list(self.runtime.onnx_providers))
        input_names = [input_info.name for input_info in session.get_inputs()]
        model_input_type = session.get_inputs()[0].type
        self.model_dtype = np.float16 if "float16" in model_input_type else np.float32

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

        self._report("audio_features", 20, "提取音频特征...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(self.hubert_path))
        hubert_model = HubertModel.from_pretrained(str(self.hubert_path)).to(self.device).eval()
        if self.runtime.resolved == "cuda":
            hubert_model = hubert_model.half()
        wav, sample_rate = sf.read(str(temp_audio_path))
        input_values = feature_extractor(wav, sampling_rate=sample_rate, return_tensors="pt").input_values
        input_values = input_values.to(self.device)
        if self.runtime.resolved == "cuda":
            input_values = input_values.half()

        with torch.no_grad():
            outputs = hubert_model(input_values)
            reps = outputs.last_hidden_state.permute(0, 2, 1).cpu().numpy()

        del hubert_model, feature_extractor
        if self.runtime.resolved == "cuda":
            torch.cuda.empty_cache()

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

        self._report("face_detection", 35, "人脸检测...")
        frame_height, frame_width = full_frames[0].shape[:-1]
        total_batches = int(np.ceil(float(len(full_frames)) / self.wav2lip_batch_size))
        writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

        try:
            for batch_index, (img_batch, mel_batch, frames, coords, affines) in enumerate(
                tqdm(
                    self._datagen(full_frames, rep_chunks),
                    total=total_batches,
                    desc="Generating video",
                )
            ):
                pct = 40 + int(50 * batch_index / max(total_batches, 1))
                self._report("inference", pct, f"推理中 {batch_index * self.wav2lip_batch_size}/{len(rep_chunks)}")
                mel_batch = np.transpose(mel_batch, (0, 2, 1))
                generated_frames = []
                for frame_index in range(mel_batch.shape[0]):
                    if (batch_index == 0 and frame_index == 0) or (
                        (batch_index * self.wav2lip_batch_size + frame_index + 1) % self.syncnet_T == 0
                    ):
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

                prediction = np.stack(generated_frames, axis=0)
                for pred, frame, coords_item, affine in zip(prediction, frames, coords, affines):
                    x1, y1, x2, y2 = coords_item
                    pred = pred[[2, 1, 0], :, :]
                    resized = resize_tensor_image(
                        torch.from_numpy(pred).to(self.device),
                        size=(int(y2 - y1), int(x2 - x1)),
                    )
                    merged = self.detect_face.restorer.restore_img(frame, resized, affine, scale_h=self.scale_h, scale_w=self.scale_w)
                    writer.write(merged)

                if (batch_index + 1) % 3 == 0:
                    gc.collect()
                    if self.runtime.resolved == "cuda":
                        torch.cuda.empty_cache()
        finally:
            writer.release()

        self._report("compositing", 92, "合成视频...")
        try:
            if not self._try_nvenc_final_merge(temp_video_path, temp_audio_path, final_video_path):
                run_command(
                    [
                        self.ffmpeg_bin,
                        "-y",
                        "-i",
                        str(temp_video_path),
                        "-i",
                        str(temp_audio_path),
                        "-c:v",
                        "libx264",
                        "-crf",
                        "16",
                        "-preset",
                        "superfast",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        "-shortest",
                        str(final_video_path),
                    ]
                )
        finally:
            gc.collect()
            if self.runtime.resolved == "cuda":
                torch.cuda.empty_cache()

        self._report("completed", 100, "完成")
        return final_video_path

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

    def _try_nvenc_final_merge(self, video_path: Path, audio_path: Path, output_path: Path) -> bool:
        try:
            run_command(
                [
                    self.ffmpeg_bin,
                    "-y",
                    "-i",
                    str(video_path),
                    "-i",
                    str(audio_path),
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p4",
                    "-rc",
                    "vbr",
                    "-cq",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-shortest",
                    str(output_path),
                ]
            )
            return True
        except Exception:
            logger.info("NVENC not available for final merge, falling back to libx264")
            return False
