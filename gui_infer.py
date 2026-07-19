import logging
import os

# Quiet startup: no version-check/analytics requests, and no INFO logs from the
# HTTP client (slicer.py sets the root logger to INFO, which makes httpx chatty)
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
logging.getLogger('httpx').setLevel(logging.WARNING)

import numpy as np
import torch
import soundfile as sf
import click
import gradio as gr
import tempfile
import gc
import traceback
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from slicer import Slicer

from infer import (
    load_models, 
    load_audio, 
    apply_fade, 
    batch_process_segments
)

global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg, device
svc_model = vocoder = rmvpe = hubert = rms_extractor = spk2idx = dataset_cfg = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REPO_ROOT = Path(__file__).resolve().parent
train_process = None

LANGUAGES = {
    "En": {
        "app_title": "RIFT-SVC Voice Conversion",
        "tab_inference": "🎵 Inference",
        "tab_train": "🛠️ Training",
        "main_title": "🎤 RIFT-SVC Voice Conversion",
        "subtitle": "Convert singing or speech to target voice using RIFT-SVC model",
        "github_info": "🔗 <strong>Want to fine-tune your own speakers?</strong> Visit the <a href=\"https://github.com/Pur1zumu/RIFT-SVC\" target=\"_blank\">RIFT-SVC GitHub repository</a> for complete training and fine-tuning guides.",
        "audio_note": "📝 <strong>Note:</strong> For best results, use clean audio with minimal background noise.",
        
        "input_section": "### 📥 Input",
        "model_path_label": "Model Path",
        "model_path_placeholder": "Enter your model file path",
        "load_model_btn": "🔄 Load Model",
        "input_audio_label": "Input Audio File",
        
        "basic_params": "⚙️ Basic Parameters",
        "target_speaker": "Target Speaker",
        "key_shift": "Key Shift (semitones)",
        "infer_steps": "Inference Steps",
        "infer_steps_info": "Lower = faster but lower quality, Higher = slower but better quality",
        "use_fp16": "Use FP16 Precision",
        "use_fp16_info": "Enable for better performance, may reduce precision on some GPUs",
        "pitch_filter": "Pitch Filter",
        "pitch_filter_info": "0=None, 1=Light filtering, 2=Strong filtering (helps with broken/choppy sounds)",
        "batch_size": "Batch Size",
        "batch_size_info": "Number of segments to process in parallel. Higher values can be faster on GPUs with sufficient memory.",
        
        "adv_cfg_params": "🔬 Advanced CFG Parameters",
        "ds_cfg_strength": "Content Vector Guidance Strength",
        "ds_cfg_strength_info": "Higher values can improve content preservation and pronunciation clarity. Too high will be overkill.",
        "spk_cfg_strength": "Speaker Guidance Strength",
        "spk_cfg_strength_info": "Higher values can enhance speaker similarity. Too high may cause distortion.",
        "skip_cfg_strength": "Layer Guidance Strength (Experimental)",
        "skip_cfg_strength_info": "Enhances feature rendering at specified layers. Effect depends on target layer functionality.",
        "cfg_skip_layers": "CFG Skip Layers (Experimental)",
        "cfg_skip_layers_info": "Target enhancement layer index, -1 to disable this feature",
        "cfg_rescale": "CFG Rescale Factor",
        "cfg_rescale_info": "Constrains overall guidance strength. Increase when guidance effects are too strong.",
        "cvec_downsample": "Content Vector Downsample Rate for Reverse Guidance",
        "cvec_downsample_info": "Higher values (may) improve content clarity.",

        "sampling_params": "🧪 Sampling Parameters",
        "ode_method": "ODE Solver",
        "ode_method_info": "midpoint gives better quality per step than euler (each step costs 2 evaluations)",
        "sway_coef": "Sway Sampling Coefficient",
        "sway_coef_info": "Negative values concentrate steps early in the trajectory (e.g. -0.5). 0 disables.",
        "seed": "Seed",
        "seed_info": "Random seed for the initial noise. Keeps timbre consistent across segments and runs.",

        "slicer_params": "✂️ Slicer Parameters",
        "slicer_threshold": "Threshold (dB)",
        "slicer_threshold_info": "Silence detection threshold",
        "slicer_min_length": "Minimum Length (ms)",
        "slicer_min_length_info": "Minimum segment length",
        "slicer_min_interval": "Minimum Silence Interval (ms)",
        "slicer_min_interval_info": "Minimum interval for segment splitting",
        "slicer_hop_size": "Hop Size (ms)",
        "slicer_hop_size_info": "Window size for segment detection",
        "slicer_max_sil": "Maximum Silence Kept (ms)",
        "slicer_max_sil_info": "Maximum silence length kept at each segment edge",
        
        "convert_btn": "🎵 Convert Voice",
        "output_section": "### 📤 Output",
        "output_audio_label": "Converted Audio",
        "init_message": "⏳ Please load a model to begin.",
        
        "quick_tips": "🔍 Quick Tips",
        "tips_content": """
                    <ul>
                        <li><strong>Key Shift:</strong> Adjust pitch up or down in semitones.</li>
                        <li><strong>Inference Steps:</strong> More steps = better quality but slower.</li>
                        <li><strong>Pitch Filter:</strong> Helps with pitch stability in challenging audio.</li>
                        <li><strong>CFG Parameters:</strong> Adjust conversion quality and timbre.</li>
                    </ul>
                """,
        
        "processing": "⏳ Processing... Please wait.",
        "loading_audio": "Processing: Loading audio...",
        "slicing_audio": "Processing: Slicing audio...",
        "start_conversion": "Processing: Starting conversion...",
        "processing_segment": "Processing: Segment {}/{}",
        "finalizing_audio": "Processing: Finalizing audio...",
        "processing_complete": "Processing complete!",
        
        "conversion_complete": "✅ Conversion complete! Converted to **{}** with **{}** semitone shift.",
        "error_no_audio": "❌ Error: No input audio provided.",
        "error_no_model": "❌ Error: Model not loaded. Please load a model first.",
        "error_invalid_speaker": "❌ Error: Invalid speaker selection. Available speakers: {}",
        "error_no_segments": "❌ Error: No valid audio segments found in the input file.",
        "error_out_of_memory": "❌ Error: Out of memory. Try a shorter audio file or reduce inference steps.",
        "error_conversion": "❌ Error during conversion: {}",
        "error_details": "❌ Error during conversion: {}\n\nDetails: {}",
        "error_model_not_found": "❌ Error: Model not found",
        "model_loaded_success": "✅ Model loaded successfully! Available speakers: ",
        "error_loading_model": "❌ Error: Failed to load model",
        "error_details_label": "Error Details"
    },
    "中文": {
        "app_title": "RIFT-SVC 声音转换",
        "tab_inference": "🎵 推理",
        "tab_train": "🛠️ 训练",
        "main_title": "🎤 RIFT-SVC 歌声音色转换",
        "subtitle": "使用 RIFT-SVC 模型将歌声或语音转换为目标音色",
        "github_info": "🔗 <strong>想要微调自己的说话人？</strong> 请访问 <a href=\"https://github.com/Pur1zumu/RIFT-SVC\" target=\"_blank\">RIFT-SVC GitHub 仓库</a> 获取完整的训练和微调指南。",
        "audio_note": "📝 <strong>注意：</strong> 为获得最佳效果，请使用背景噪音较少的干净音频。",
        
        "input_section": "### 📥 输入",
        "model_path_label": "模型路径",
        "model_path_placeholder": "请输入您的模型文件路径",
        "load_model_btn": "🔄 加载模型",
        "input_audio_label": "输入音频文件",
        
        "basic_params": "⚙️ 基本参数",
        "target_speaker": "目标说话人",
        "key_shift": "音调调整（半音）",
        "infer_steps": "推理步数",
        "infer_steps_info": "更低的值 = 更快但质量较低，更高的值 = 更慢但质量更好",
        "use_fp16": "使用 FP16 精度",
        "use_fp16_info": "启用以提高性能，在某些GPU上可能会降低精度",
        "pitch_filter": "音高滤波",
        "pitch_filter_info": "0=无，1=轻度过滤，2=强力过滤（有助于解决断音/破音问题）",
        "batch_size": "批量大小",
        "batch_size_info": "并行处理段落的数量。更高的值可以在具有足够内存的GPU上更快。",
        
        "adv_cfg_params": "🔬 高级CFG参数",
        "ds_cfg_strength": "内容向量引导强度",
        "ds_cfg_strength_info": "更高的值可以改善内容保留和咬字清晰度。过高会用力过猛。",
        "spk_cfg_strength": "说话人引导强度",
        "spk_cfg_strength_info": "更高的值可以增强说话人相似度。过高可能导致音色失真。",
        "skip_cfg_strength": "层引导强度（实验性功能）",
        "skip_cfg_strength_info": "增强指定层的特征渲染。效果取决于目标层的功能。",
        "cfg_skip_layers": "CFG跳过层（实验性功能）",
        "cfg_skip_layers_info": "目标增强层下标，-1为禁用此功能",
        "cfg_rescale": "CFG重缩放因子",
        "cfg_rescale_info": "约束整体引导强度。当引导效果过于强烈时使用调高该值。",
        "cvec_downsample": "用于反向引导的内容向量下采样率",
        "cvec_downsample_info": "更高的值（可能）可以提高内容清晰度。",

        "sampling_params": "🧪 采样参数",
        "ode_method": "ODE 求解器",
        "ode_method_info": "midpoint 每步质量优于 euler（每步需要2次前向计算）",
        "sway_coef": "Sway 采样系数",
        "sway_coef_info": "负值将采样步集中在轨迹前段（如 -0.5）。0 为禁用。",
        "seed": "随机种子",
        "seed_info": "初始噪声的随机种子。保持各片段和多次运行之间音色一致。",

        "slicer_params": "✂️ 切片参数",
        "slicer_threshold": "阈值 (dB)",
        "slicer_threshold_info": "静音检测阈值",
        "slicer_min_length": "最小长度 (毫秒)",
        "slicer_min_length_info": "最小片段长度",
        "slicer_min_interval": "最小静音间隔 (毫秒)",
        "slicer_min_interval_info": "分割片段的最小间隔",
        "slicer_hop_size": "跳跃大小 (毫秒)",
        "slicer_hop_size_info": "片段检测窗口大小",
        "slicer_max_sil": "保留的最大静音 (毫秒)",
        "slicer_max_sil_info": "保留在每个片段边缘的最大静音长度",
        
        "convert_btn": "🎵 转换声音",
        "output_section": "### 📤 输出",
        "output_audio_label": "转换后的音频",
        "init_message": "⏳ 请加载模型以开始使用。",
        
        "quick_tips": "🔍 快速提示",
        "tips_content": """
                    <ul>
                        <li><strong>音调调整：</strong> 以半音为单位上调或下调音高。</li>
                        <li><strong>推理步骤：</strong> 步骤越多 = 质量越好但速度越慢。</li>
                        <li><strong>音高滤波：</strong> 有助于提高具有挑战性的音频中的音高稳定性。</li>
                        <li><strong>CFG参数：</strong> 调整转换质量和音色。</li>
                    </ul>
                """,
        
        "processing": "⏳ 处理中... 请稍候。",
        "loading_audio": "处理中: 加载音频...",
        "slicing_audio": "处理中: 切分音频...",
        "start_conversion": "处理中: 开始转换...",
        "processing_segment": "处理中: 片段 {}/{}",
        "finalizing_audio": "处理中: 完成音频...",
        "processing_complete": "处理完成!",
        
        "conversion_complete": "✅ 转换完成! 已转换为 **{}** 并调整 **{}** 个半音。",
        "error_no_audio": "❌ 错误: 未提供输入音频。",
        "error_no_model": "❌ 错误: 模型未加载。请先加载模型。",
        "error_invalid_speaker": "❌ 错误: 无效的说话人选择。可用说话人: {}",
        "error_no_segments": "❌ 错误: 在输入文件中未找到有效的音频片段。",
        "error_out_of_memory": "❌ 错误: 内存不足。请尝试更短的音频文件或减少推理步骤。",
        "error_conversion": "❌ 转换过程中出错: {}",
        "error_details": "❌ 转换过程中出错: {}\n\n详细信息: {}",
        "error_model_not_found": "❌ 错误: 找不到模型文件",
        "model_loaded_success": "✅ 模型加载成功！可用说话人: ",
        "error_loading_model": "❌ 错误: 加载模型出错",
        "error_details_label": "错误详细信息"
    }
}

current_language = "En"

def initialize_models(model_path):
    global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg, current_language

    lang = LANGUAGES[current_language]

    if svc_model is not None:
        # Release old models before loading new ones; keep globals defined so a
        # failed reload doesn't leave dangling references
        svc_model = vocoder = rmvpe = hubert = rms_extractor = None
        torch.cuda.empty_cache()
        gc.collect()

    try:
        if not os.path.exists(model_path):
            return [], f"{lang['error_model_not_found']}: {model_path}"

        # Load weights in fp32 so the fp16 checkbox (autocast only) can be toggled
        # freely at inference time without dtype mismatches
        svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg = load_models(model_path, device, use_fp16=False)
        available_speakers = list(spk2idx.keys())
        return available_speakers, f"{lang['model_loaded_success']} {', '.join(available_speakers)}"
    except Exception as e:
        error_trace = traceback.format_exc()
        return [], f"{lang['error_loading_model']}: {str(e)}\n\n{lang['error_details_label']}: {error_trace}"

def process_with_progress(
    progress=gr.Progress(),
    input_audio=None,
    speaker=None,
    key_shift=0,
    infer_steps=32,
    robust_f0=1,
    use_fp16=True,
    batch_size=1,
    ds_cfg_strength=0.1,
    spk_cfg_strength=1.0,
    skip_cfg_strength=0.0,
    cfg_skip_layers=6,
    cfg_rescale=0.7,
    cvec_downsample_rate=2,
    slicer_threshold=-30.0,
    slicer_min_length=3000,
    slicer_min_interval=100,
    slicer_hop_size=10,
    slicer_max_sil_kept=200,
    ode_method='euler',
    sway_coef=0.0,
    seed=42
):
    global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg, current_language
    
    lang = LANGUAGES[current_language]
    
    target_loudness = -18.0
    restore_loudness = True
    fade_duration = 20.0
    
    if input_audio is None:
        return None, lang["error_no_audio"]
    
    if svc_model is None:
        return None, lang["error_no_model"]
    
    if speaker is None or speaker not in spk2idx:
        return None, lang["error_invalid_speaker"].format(", ".join(spk2idx.keys()))
    
    try:
        progress(0, desc=lang["loading_audio"])
        
        speaker_id = spk2idx[speaker]
        
        hop_length = 512
        sample_rate = 44100
        
        if cfg_skip_layers is None or int(cfg_skip_layers) < 0:
            cfg_skip_layers_value = None
        else:
            cfg_skip_layers_value = int(cfg_skip_layers)

        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        
        audio = load_audio(input_audio, sample_rate)
        
        slicer = Slicer(
            sr=sample_rate,
            threshold=slicer_threshold,
            min_length=slicer_min_length,
            min_interval=slicer_min_interval,
            hop_size=slicer_hop_size,
            max_sil_kept=slicer_max_sil_kept
        )
        
        progress(0.1, desc=lang["slicing_audio"])
        segments_with_pos = slicer.slice(audio)
        
        if not segments_with_pos:
            return None, lang["error_no_segments"]
        
        fade_samples = int(fade_duration * sample_rate / 1000)
        
        progress(0.2, desc=lang["start_conversion"])
        
        with torch.no_grad():
            processed_segments = batch_process_segments(
                segments_with_pos, svc_model, vocoder, rmvpe, hubert, rms_extractor,
                speaker_id, sample_rate, hop_length, device,
                key_shift, infer_steps, ds_cfg_strength, spk_cfg_strength,
                skip_cfg_strength, cfg_skip_layers_value, cfg_rescale,
                cvec_downsample_rate, target_loudness, restore_loudness,
                robust_f0, use_fp16, batch_size, progress, lang["processing_segment"],
                ode_method=ode_method, sway_coef=sway_coef, generator=generator
            )
            
            result_audio = np.zeros(len(audio) + fade_samples)
            
            for idx, (start_sample, audio_out, expected_length) in enumerate(processed_segments):
                segment_progress = 0.8 + (0.1 * (idx / len(processed_segments)))
                progress(segment_progress, desc=lang["finalizing_audio"])
                
                if len(audio_out) > expected_length:
                    audio_out = audio_out[:expected_length]
                elif len(audio_out) < expected_length:
                    audio_out = np.pad(audio_out, (0, expected_length - len(audio_out)), 'constant')
                
                if idx > 0:
                    audio_out = apply_fade(audio_out.copy(), fade_samples, fade_in=True)
                    result_audio[start_sample:start_sample + fade_samples] *= np.linspace(1, 0, fade_samples)
                
                if idx < len(processed_segments) - 1:
                    audio_out[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                result_audio[start_sample:start_sample + len(audio_out)] += audio_out
        
        progress(0.9, desc=lang["finalizing_audio"])
        result_audio = result_audio[:len(audio)]
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            output_path = temp_file.name
        
        sf.write(output_path, result_audio.astype(np.float32), sample_rate)
        
        progress(1.0, desc=lang["processing_complete"])
        return (sample_rate, result_audio), lang["conversion_complete"].format(speaker, key_shift)
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            gc.collect()
            
            return None, lang["error_out_of_memory"]
        else:
            return None, lang["error_conversion"].format(str(e))
    except Exception as e:
        error_trace = traceback.format_exc()
        return None, lang["error_details"].format(str(e), error_trace)
    finally:
        torch.cuda.empty_cache()
        gc.collect()

# ------------------------- Training tab backend -------------------------

DEFAULT_DATASET_DIR = "assets/dataset"
DEFAULT_TRAIN_DATA_DIR = "data/finetune"


def _tail(lines, n=120):
    return "\n".join(lines[-n:])


def _stream_command(cmd, cwd=None, process_slot=None):
    """Run a command yielding decoded output chunks as they arrive.

    Reads the pipe with os.read so tqdm-style carriage-return updates are
    visible without waiting for newlines.
    """
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    proc = subprocess.Popen(
        cmd, cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env,
    )
    if process_slot is not None:
        process_slot.append(proc)
    try:
        while True:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            yield chunk.decode('utf-8', errors='replace')
    finally:
        proc.stdout.close()
        proc.wait()
    yield None if proc.returncode == 0 else f"\n[exit code {proc.returncode}]"


def _append_output(lines, text):
    """Merge raw subprocess output into a list of lines, honoring \\r updates."""
    for piece in re.split(r'(\r\n|\n|\r)', text):
        if piece in ('\n', '\r\n'):
            lines.append('')
        elif piece == '\r':
            if lines and lines[-1]:
                lines[-1] = ''
        elif piece:
            if not lines:
                lines.append('')
            lines[-1] += piece


AUDIO_EXTS = ('.wav', '.flac')


def _has_audio(d):
    return any(f.suffix.lower() in AUDIO_EXTS for f in d.rglob('*'))


def _find_speaker_dirs(root):
    """Speaker folders under root; if root itself holds the audio files directly,
    treat root as a single speaker named after the folder."""
    root = Path(root)
    if not root.is_dir():
        return []
    speakers = [d for d in sorted(root.iterdir()) if d.is_dir() and _has_audio(d)]
    if not speakers and any(f.suffix.lower() in AUDIO_EXTS for f in root.iterdir() if f.is_file()):
        speakers = [root]
    return speakers


def scan_dataset(dataset_dir):
    """Return a markdown summary of the speakers found in dataset_dir."""
    root = Path(dataset_dir)
    if not root.is_dir():
        return f"❌ Directory not found: `{dataset_dir}`"
    speakers = _find_speaker_dirs(root)
    if not speakers:
        return (f"❌ No .wav/.flac files found in `{dataset_dir}`.\n\n"
                "Expected layout: `assets/dataset/<speaker_name>/*.wav` "
                "(or point directly at a folder containing the audio files "
                "to treat it as a single speaker)")
    md = f"✅ Found **{len(speakers)}** speaker(s) in `{dataset_dir}`:\n\n"
    md += "\n".join(
        f"- **{d.name}**: {len([f for f in d.rglob('*') if f.suffix.lower() in AUDIO_EXTS])} file(s)"
        for d in speakers
    )
    if speakers == [root]:
        md += f"\n\n(ℹ️ audio files found directly in the folder — using `{root.name}` as the speaker name)"
    return md


def _slice_audio_file(f, out_dir, threshold, min_len_ms, max_len_s, overwrite):
    """Slice one audio file on silence and write the segments as wav files.

    Segments longer than max_len_s are split into equal parts; leftovers shorter
    than 1s are dropped. Returns (n_written, n_skipped_files).
    """
    first_out = out_dir / f'{f.stem}_000.wav'
    if first_out.exists() and not overwrite:
        return 0, 1

    data, sr = sf.read(str(f), dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)

    slicer = Slicer(sr=sr, threshold=threshold, min_length=int(min_len_ms),
                    min_interval=100, hop_size=20, max_sil_kept=500)
    segments = [chunk for _, chunk in slicer.slice(data)]

    max_samples = int(max_len_s * sr)
    pieces = []
    for chunk in segments:
        if len(chunk) <= max_samples:
            pieces.append(chunk)
        else:
            n_parts = math.ceil(len(chunk) / max_samples)
            part_len = math.ceil(len(chunk) / n_parts)
            pieces.extend(chunk[i * part_len:(i + 1) * part_len] for i in range(n_parts))

    n_written = 0
    for chunk in pieces:
        if len(chunk) < sr:  # drop segments shorter than 1 second
            continue
        sf.write(str(out_dir / f'{f.stem}_{n_written:03d}.wav'), chunk, sr)
        n_written += 1
    return n_written, 0


def run_preprocess(dataset_dir, data_dir, num_workers, overwrite, num_test_per_speaker,
                   slice_enable=True, slice_threshold=-40.0, slice_min_len_ms=3000, slice_max_len_s=15.0):
    """Copy audio from dataset_dir into data_dir and run the full feature pipeline."""
    lines = []

    def emit(msg=""):
        lines.append(msg)
        return _tail(lines)

    src_root = Path(dataset_dir)
    dst_root = Path(data_dir)
    if not src_root.is_dir():
        yield emit(f"ERROR: dataset directory not found: {dataset_dir}")
        return

    speakers = _find_speaker_dirs(src_root)
    if not speakers:
        yield emit(f"ERROR: no audio (.wav/.flac) found in {dataset_dir}")
        return
    if speakers == [src_root]:
        yield emit(f"Audio found directly in the folder — treating it as single speaker '{src_root.name}'.")

    # 1. Copy (and optionally slice) audio into the training data dir so the
    # originals in assets/dataset are never modified by normalization.
    if slice_enable:
        yield emit(f"[1/6] Slicing audio on silence (threshold {slice_threshold} dB, "
                   f"min {int(slice_min_len_ms)} ms, max {slice_max_len_s:g} s) into {data_dir} ...")
    else:
        yield emit(f"[1/6] Copying audio from {dataset_dir} to {data_dir} ...")
    n_copied, n_skipped = 0, 0
    for spk_dir in speakers:
        out_dir = dst_root / spk_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(spk_dir.rglob('*')):
            if f.suffix.lower() not in ('.wav', '.flac'):
                continue
            try:
                if slice_enable:
                    written, skipped = _slice_audio_file(
                        f, out_dir, slice_threshold, slice_min_len_ms, slice_max_len_s, overwrite)
                    n_copied += written
                    n_skipped += skipped
                else:
                    out_path = out_dir / (f.stem + '.wav')
                    if out_path.exists() and not overwrite:
                        n_skipped += 1
                        continue
                    if f.suffix.lower() == '.flac':
                        data, sr = sf.read(str(f), always_2d=False)
                        sf.write(str(out_path), data, sr)
                    else:
                        shutil.copy2(f, out_path)
                    n_copied += 1
            except Exception as e:
                yield emit(f"    WARNING: failed to process {f.name}: {e}")
        yield emit(f"    {spk_dir.name}: done")
    if slice_enable:
        yield emit(f"    Wrote {n_copied} segment(s), skipped {n_skipped} already-sliced file(s).")
    else:
        yield emit(f"    Copied {n_copied} file(s), skipped {n_skipped} existing.")
    if n_copied == 0 and n_skipped == 0:
        yield emit("ERROR: no audio was produced. Check the slicer threshold/lengths and the input files.")
        return

    stages = [
        ("[2/6] Resampling to 44100 Hz and normalizing to -18 LUFS",
         [sys.executable, 'scripts/resample_normalize_audios.py', '--src', str(dst_root)]),
        ("[3/6] Generating meta_info.json",
         [sys.executable, 'scripts/prepare_data_meta.py', '--data-dir', str(dst_root),
          '--split-type', 'stratified', '--num-test-per-speaker', str(int(num_test_per_speaker))]),
        ("[4/6] Extracting mel spectrograms",
         [sys.executable, 'scripts/prepare_mel.py', '--data-dir', str(dst_root),
          '--num-workers', str(int(num_workers))] + (['--overwrite'] if overwrite else [])),
        ("[4/6] Extracting RMS",
         [sys.executable, 'scripts/prepare_rms.py', '--data-dir', str(dst_root),
          '--num-workers', str(int(num_workers))]),
        ("[5/6] Extracting F0 (RMVPE)",
         [sys.executable, 'scripts/prepare_f0.py', '--data-dir', str(dst_root),
          '--num-workers', str(int(num_workers))] + (['--overwrite'] if overwrite else [])),
        ("[6/6] Extracting content vectors (ContentVec)",
         [sys.executable, 'scripts/prepare_cvec.py', '--data-dir', str(dst_root),
          '--num-workers', str(int(num_workers))] + (['--overwrite'] if overwrite else [])),
    ]

    for title, cmd in stages:
        yield emit(f"{title} ...")
        failed = False
        for chunk in _stream_command(cmd):
            if chunk is None:
                continue
            if chunk.startswith('\n[exit code'):
                failed = True
            _append_output(lines, chunk)
            yield _tail(lines)
        if failed:
            yield emit("ERROR: stage failed, aborting preprocessing.")
            return

    yield emit("✅ Preprocessing complete! The dataset is ready for training.")


def count_dataset_speakers(train_data_dir, dataset_dir):
    """Number of speakers, from meta_info.json if preprocessed, else from raw subfolders."""
    meta_path = Path(train_data_dir) / 'meta_info.json'
    if meta_path.exists():
        try:
            import json
            with open(meta_path, 'r', encoding='utf-8') as f:
                return len(json.load(f).get('speakers', [])), 'meta_info.json'
        except Exception:
            pass
    root = Path(dataset_dir)
    if root.is_dir():
        return len(_find_speaker_dirs(root)), 'dataset folders'
    return 0, None


def suggest_speaker_settings(train_data_dir, dataset_dir):
    """Auto-set freeze_adaln + drop_spk_prob from the number of speakers.

    Single speaker: freeze adaLN/tembed and never drop the speaker, preserving the
    pretrained null-speaker branch for inference-time speaker CFG. Multi-speaker:
    unfreeze and drop at 0.2 so the unconditional branch trains along (README recipe).
    """
    n, source = count_dataset_speakers(train_data_dir, dataset_dir)
    if n == 0:
        return gr.update(), gr.update(), ""
    if n == 1:
        note = (f"🎯 **1 speaker** detected ({source}): using the single-speaker recipe — "
                "freeze adaLN/time-embedding **on**, speaker dropout **0.0**.")
        return gr.update(value=True), gr.update(value=0.0), note
    note = (f"🎯 **{n} speakers** detected ({source}): using the multi-speaker recipe — "
            "freeze adaLN/time-embedding **off**, speaker dropout **0.2**.")
    return gr.update(value=False), gr.update(value=0.2), note


def list_pretrained_ckpts():
    ckpts = sorted((REPO_ROOT / 'pretrained').glob('*.ckpt'))
    return [str(p.relative_to(REPO_ROOT).as_posix()) for p in ckpts]


def list_resume_ckpts():
    """Checkpoints from previous runs (ckpts/<run_name>/*.ckpt), newest first."""
    ckpts = sorted((REPO_ROOT / 'ckpts').glob('*/*.ckpt'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [''] + [str(p.relative_to(REPO_ROOT).as_posix()) for p in ckpts]


def guess_model_from_ckpt(ckpt_path):
    m = re.search(r'dit-\d+-\d+', str(ckpt_path or ''))
    if m:
        cfg = REPO_ROOT / 'config' / 'model' / f'{m.group(0)}.yaml'
        if cfg.exists():
            return gr.update(value=m.group(0))
    return gr.update()


def start_training(
    data_dir, model_name, pretrained_ckpt, run_name,
    learning_rate, weight_decay, max_steps, warmup_ratio,
    batch_size, max_frame_len, save_per_steps, test_per_steps,
    drop_spk_prob, ema_decay, max_grad_norm,
    freeze_adaln, num_workers,
    grad_accum=1, compile_model=False,
    resume_ckpt='', save_full_ckpt=True,
):
    global train_process
    if train_process is not None and train_process.poll() is None:
        yield "⚠️ A training run is already in progress. Stop it first."
        return

    data_dir = str(Path(data_dir).as_posix())
    meta_path = Path(data_dir) / 'meta_info.json'
    if not meta_path.exists():
        yield f"ERROR: {meta_path} not found. Run preprocessing first."
        return

    resume_ckpt = (resume_ckpt or '').strip()
    if resume_ckpt:
        rp = Path(resume_ckpt)
        if not rp.is_absolute():
            rp = REPO_ROOT / rp
        if not rp.exists():
            yield f"ERROR: resume checkpoint not found: {rp}"
            return
        yield "Checking resume checkpoint..."
        try:
            ck = torch.load(str(rp), map_location='cpu', mmap=True, weights_only=False)
            resumable = 'optimizer_states' in ck
            del ck
        except Exception as e:
            yield f"ERROR: could not read resume checkpoint: {e}"
            return
        if not resumable:
            yield ("ERROR: this is a weights-only checkpoint and cannot be resumed "
                   "(no optimizer state). Train with 'Save full checkpoints' enabled "
                   "to produce resumable checkpoints — or use it as 'Pretrained Checkpoint' "
                   "instead to start a new run from those weights.")
            return
        resume_ckpt = rp.as_posix()

    run_name = re.sub(r'[^\w\-.]+', '_', run_name.strip()) or 'finetune_gui'

    overrides = [
        f'model={model_name}',
        f'dataset.data_dir={data_dir}',
        f'dataset.meta_info_path={data_dir}/meta_info.json',
        f'dataset.max_frame_len={int(max_frame_len)}',
        f'training.run_name={run_name}',
        f'training.learning_rate={learning_rate}',
        f'training.weight_decay={weight_decay}',
        f'training.max_steps={int(max_steps)}',
        f'training.warmup_ratio={warmup_ratio}',
        f'training.batch_size_per_gpu={int(batch_size)}',
        f'training.save_per_steps={int(save_per_steps)}',
        f'training.test_per_steps={int(test_per_steps)}',
        f'training.log_media_per_steps={int(test_per_steps)}',
        f'training.drop_spk_prob={drop_spk_prob}',
        f'training.ema_decay={ema_decay}',
        f'training.max_grad_norm={max_grad_norm}',
        f'training.num_workers={int(num_workers)}',
        f'training.grad_accumulation_steps={max(1, int(grad_accum))}',
        f'training.compile_model={"true" if compile_model else "false"}',
        f'training.save_weights_only={"false" if save_full_ckpt else "true"}',
        'training.logger=tensorboard',
    ]
    if resume_ckpt:
        overrides.append(f'+training.resume_from_checkpoint={resume_ckpt}')
    overrides.append(f'training.freeze_adaln_and_tembed={"true" if freeze_adaln else "false"}')
    if pretrained_ckpt:
        overrides.append(f'+training.pretrained_path={Path(pretrained_ckpt).as_posix()}')

    cmd = [sys.executable, 'train.py', '--config-name', 'finetune'] + overrides

    lines = [
        "Launching training:",
        "  " + " ".join(cmd),
        f"Checkpoints: ckpts/{run_name}/  |  TensorBoard logs: logs/{run_name}/",
        f"  (view with: tensorboard --logdir logs/{run_name})",
        "",
    ]
    yield _tail(lines)

    slot = []
    try:
        for chunk in _stream_command(cmd, process_slot=slot):
            if slot and train_process is not slot[0]:
                train_process = slot[0]
            if chunk is None:
                continue
            _append_output(lines, chunk)
            yield _tail(lines)
    finally:
        proc = slot[0] if slot else None
        if proc is not None and proc.poll() is None:
            proc.terminate()

    code = slot[0].returncode if slot else -1
    lines.append("")
    if code == 0:
        lines.append("✅ Training finished!")
        run_dir = REPO_ROOT / 'ckpts' / run_name
        newest = max(run_dir.glob('*.ckpt'), key=lambda p: p.stat().st_mtime, default=None)
        if newest is not None:
            lines.append(f"Latest checkpoint: ckpts/{run_name}/{newest.name}")
        if (run_dir / 'model-best.ckpt').exists():
            lines.append(f"Best by val/mcd: ckpts/{run_name}/model-best.ckpt")
    else:
        lines.append(f"❌ Training exited with code {code} (or was stopped).")
    yield _tail(lines)


def stop_training():
    global train_process
    if train_process is not None and train_process.poll() is None:
        train_process.terminate()
        return "🛑 Stop signal sent to the training process."
    return "No training process is running."


def create_ui():
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        position: relative;
    }
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .footer {
        margin-top: 20px;
        text-align: center;
        font-size: 0.9em;
        color: var(--body-text-color-subdued);
    }
    .title {
        text-align: center;
        margin-bottom: 10px;
        padding-top: 10px;
    }
    .subtitle {
        text-align: center;
        margin-bottom: 20px;
        color: var(--body-text-color-subdued);
    }
    .button-primary {
        background-color: #5460DE !important;
    }
    .output-message {
        margin-top: 10px;
        padding: 10px;
        border-radius: 4px;
        background-color: var(--background-fill-secondary);
        color: var(--body-text-color);
        border-left: 4px solid #5460DE;
    }
    .error-message {
        color: #e05252;
        font-weight: bold;
    }
    .success-message {
        color: #6aa87c;
        font-weight: bold;
    }
    .info-box {
        background-color: var(--background-fill-secondary);
        color: var(--body-text-color);
        border-left: 4px solid #5460DE;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .info-box a {
        color: var(--link-text-color, #5460DE);
    }
    .lang-container {
        position: absolute;
        top: 10px;
        right: 20px;
        z-index: 1000;
        width: auto;
        max-width: 180px;
    }
    .compact-dropdown .wrap .label-wrap {
        font-size: 0.9em !important;
    }
    .compact-dropdown .wrap {
        max-width: 150px !important;
    }
    """
    
    available_speakers = []
    global current_language
    lang = LANGUAGES[current_language]
    init_message = lang["init_message"]
    
    with gr.Blocks(css=css, theme=gr.themes.Soft(), title=lang["app_title"]) as app:
        with gr.Row(elem_classes="lang-container"):
            language_selector = gr.Dropdown(
                choices=["En", "中文"], 
                value=current_language, 
                label="Language / 语言",
                elem_classes="compact-dropdown"
            )
        
        html_header = gr.HTML(f"""
        <div class="title">
            <h1>{lang["main_title"]}</h1>
        </div>
        <div class="subtitle">
            <h3>{lang["subtitle"]}</h3>
        </div>
        <div class="info-box">
            <p>{lang["github_info"]}</p>
        </div>
        <div class="info-box">
            <p>{lang["audio_note"]}</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.TabItem(lang["tab_inference"], id="tab_inference") as inference_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            input_markdown = gr.Markdown(lang["input_section"])
                            model_path = gr.Textbox(label=lang["model_path_label"], value="", placeholder=lang["model_path_placeholder"], interactive=True)
                            reload_btn = gr.Button(lang["load_model_btn"], elem_id="reload_btn")
                            input_audio = gr.Audio(label=lang["input_audio_label"], type="filepath", elem_id="input_audio")

                        with gr.Accordion(lang["basic_params"], open=True) as basic_params_accordion:
                            speaker = gr.Dropdown(label=lang["target_speaker"], interactive=True, elem_id="speaker")
                            key_shift = gr.Slider(minimum=-12, maximum=12, step=1, value=0, label=lang["key_shift"], elem_id="key_shift")
                            infer_steps = gr.Slider(minimum=8, maximum=64, step=1, value=32, label=lang["infer_steps"], elem_id="infer_steps",
                                                   info=lang["infer_steps_info"])
                            use_fp16 = gr.Checkbox(label=lang["use_fp16"], value=True, info=lang["use_fp16_info"], elem_id="use_fp16")
                            robust_f0 = gr.Radio(choices=[0, 1, 2], value=1, label=lang["pitch_filter"],
                                                info=lang["pitch_filter_info"],
                                                elem_id="robust_f0")
                            batch_size = gr.Slider(minimum=1, maximum=64, step=1, value=1, label=lang["batch_size"], info=lang["batch_size_info"], elem_id="batch_size")

                        with gr.Accordion(lang["sampling_params"], open=True) as sampling_accordion:
                            ode_method = gr.Dropdown(choices=['euler', 'midpoint', 'rk4'], value='euler',
                                                    label=lang["ode_method"],
                                                    info=lang["ode_method_info"],
                                                    elem_id="ode_method")
                            sway_coef = gr.Slider(minimum=-1.0, maximum=1.0, step=0.05, value=0.0,
                                                 label=lang["sway_coef"],
                                                 info=lang["sway_coef_info"],
                                                 elem_id="sway_coef")
                            seed = gr.Number(value=42, label=lang["seed"], precision=0,
                                            info=lang["seed_info"],
                                            elem_id="seed")

                        with gr.Accordion(lang["adv_cfg_params"], open=True) as adv_cfg_accordion:
                            ds_cfg_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.1,
                                                       label=lang["ds_cfg_strength"],
                                                       info=lang["ds_cfg_strength_info"],
                                                       elem_id="ds_cfg_strength")
                            spk_cfg_strength = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=1.0,
                                                        label=lang["spk_cfg_strength"],
                                                        info=lang["spk_cfg_strength_info"],
                                                        elem_id="spk_cfg_strength")
                            skip_cfg_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                                                         label=lang["skip_cfg_strength"],
                                                         info=lang["skip_cfg_strength_info"],
                                                         elem_id="skip_cfg_strength")
                            cfg_skip_layers = gr.Number(value=-1, label=lang["cfg_skip_layers"], precision=0,
                                                       info=lang["cfg_skip_layers_info"],
                                                       elem_id="cfg_skip_layers")
                            cfg_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.7,
                                                   label=lang["cfg_rescale"],
                                                   info=lang["cfg_rescale_info"],
                                                   elem_id="cfg_rescale")
                            cvec_downsample_rate = gr.Radio(choices=[1, 2, 4, 8], value=2,
                                                          label=lang["cvec_downsample"],
                                                          info=lang["cvec_downsample_info"],
                                                          elem_id="cvec_downsample_rate")

                        with gr.Accordion(lang["slicer_params"], open=False) as slicer_accordion:
                            slicer_threshold = gr.Slider(minimum=-60.0, maximum=-20.0, step=0.1, value=-30.0,
                                                        label=lang["slicer_threshold"],
                                                        info=lang["slicer_threshold_info"],
                                                        elem_id="slicer_threshold")
                            slicer_min_length = gr.Slider(minimum=1000, maximum=10000, step=100, value=3000,
                                                         label=lang["slicer_min_length"],
                                                         info=lang["slicer_min_length_info"],
                                                         elem_id="slicer_min_length")
                            slicer_min_interval = gr.Slider(minimum=10, maximum=500, step=10, value=100,
                                                           label=lang["slicer_min_interval"],
                                                           info=lang["slicer_min_interval_info"],
                                                           elem_id="slicer_min_interval")
                            slicer_hop_size = gr.Slider(minimum=1, maximum=20, step=1, value=10,
                                                      label=lang["slicer_hop_size"],
                                                      info=lang["slicer_hop_size_info"],
                                                      elem_id="slicer_hop_size")
                            slicer_max_sil_kept = gr.Slider(minimum=10, maximum=1000, step=10, value=200,
                                                          label=lang["slicer_max_sil"],
                                                          info=lang["slicer_max_sil_info"],
                                                          elem_id="slicer_max_sil_kept")

                    with gr.Column(scale=1):
                        convert_btn = gr.Button(lang["convert_btn"], variant="primary", elem_id="convert_btn")
                        output_markdown = gr.Markdown(lang["output_section"])
                        output_audio = gr.Audio(label=lang["output_audio_label"], elem_id="output_audio", autoplay=False, show_share_button=False)
                        output_message = gr.Markdown(init_message, elem_id="output_message", elem_classes="output-message")

                        tips_html = gr.HTML(f"""
                        <div class="info-box">
                            <h4>{lang["quick_tips"]}</h4>
                            {lang["tips_content"]}
                        </div>
                        """)

            with gr.TabItem(lang["tab_train"], id="tab_train") as train_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📦 1. Dataset & Preprocessing")
                        gr.Markdown(
                            f"Place your audio as `{DEFAULT_DATASET_DIR}/<speaker_name>/*.wav` (or .flac). "
                            "Each subfolder becomes one speaker. Files are copied to the training data "
                            "directory, resampled to 44100 Hz, normalized to -18 LUFS, and all features "
                            "(mel, RMS, F0, ContentVec) are extracted."
                        )
                        dataset_dir = gr.Textbox(label="Raw Dataset Directory", value=DEFAULT_DATASET_DIR)
                        train_data_dir = gr.Textbox(label="Training Data Directory", value=DEFAULT_TRAIN_DATA_DIR,
                                                   info="Where preprocessed copies and features are stored")
                        scan_btn = gr.Button("🔍 Scan Dataset")
                        dataset_summary = gr.Markdown("")
                        with gr.Row():
                            preprocess_workers = gr.Slider(minimum=1, maximum=16, step=1, value=2,
                                                          label="Preprocess Workers",
                                                          info="Parallel processes for feature extraction")
                            num_test_per_speaker = gr.Slider(minimum=1, maximum=5, step=1, value=1,
                                                            label="Validation Files per Speaker")
                        preprocess_overwrite = gr.Checkbox(label="Overwrite existing files/features", value=False)
                        with gr.Accordion("✂️ Slicer", open=False):
                            slice_enable = gr.Checkbox(label="Slice long audios on silence", value=True,
                                                      info="Recommended: splits recordings into training-sized segments")
                            slice_threshold = gr.Slider(minimum=-60.0, maximum=-20.0, step=1.0, value=-40.0,
                                                       label="Silence Threshold (dB)")
                            slice_min_len_ms = gr.Slider(minimum=1000, maximum=10000, step=500, value=3000,
                                                        label="Minimum Segment Length (ms)")
                            slice_max_len_s = gr.Slider(minimum=5.0, maximum=30.0, step=1.0, value=15.0,
                                                       label="Maximum Segment Length (s)",
                                                       info="Longer segments are split into equal parts")
                        preprocess_btn = gr.Button("⚙️ Preprocess Dataset", variant="primary")
                        preprocess_log = gr.Textbox(label="Preprocessing Log", lines=16, max_lines=16,
                                                   interactive=False, autoscroll=True)

                    with gr.Column(scale=1):
                        gr.Markdown("### 🚀 2. Training")
                        pretrained_ckpt = gr.Dropdown(choices=list_pretrained_ckpts(),
                                                     value=(list_pretrained_ckpts() or [None])[0],
                                                     label="Pretrained Checkpoint",
                                                     info="Base model to finetune from (pretrained/*.ckpt)",
                                                     allow_custom_value=True)
                        model_name = gr.Dropdown(choices=['dit-768-12', 'dit-1024-16', 'dit-512-8'],
                                                value='dit-768-12', label="Model Architecture",
                                                info="Must match the pretrained checkpoint")
                        run_name = gr.Textbox(label="Run Name", value="finetune_gui",
                                             info="Checkpoints go to ckpts/<run_name>/, logs to logs/<run_name>/")
                        with gr.Row():
                            learning_rate = gr.Number(value=5e-5, label="Learning Rate")
                            weight_decay = gr.Number(value=0.01, label="Weight Decay")
                        with gr.Row():
                            max_steps = gr.Number(value=10000, label="Max Steps", precision=0)
                            warmup_ratio = gr.Number(value=0.05, label="Warmup Ratio")
                        with gr.Row():
                            train_batch_size = gr.Number(value=16, label="Batch Size per GPU", precision=0)
                            max_frame_len = gr.Radio(choices=[256, 384, 512, 768], value=256, label="Max Frame Length",
                                                       info="Context per sample: 256≈3s (pretrain default, lowest VRAM), 384≈4.5s, 512≈6s (recommended), 768≈9s (high VRAM). "
                                                            "On OOM, first try 384 or 256, or reduce the batch size")
                        with gr.Row():
                            save_per_steps = gr.Number(value=1000, label="Save Every N Steps", precision=0)
                            test_per_steps = gr.Number(value=1000, label="Validate Every N Steps", precision=0)
                        with gr.Accordion("Advanced", open=False):
                            with gr.Row():
                                drop_spk_prob = gr.Slider(minimum=0.0, maximum=0.5, step=0.05, value=0.0,
                                                         label="Speaker Dropout Prob",
                                                         info="0.0 for single speaker, 0.2 for multi-speaker (auto-set on scan/preprocess)")
                                ema_decay = gr.Number(value=0.999, label="EMA Decay", info="0 disables EMA")
                            with gr.Row():
                                max_grad_norm = gr.Number(value=1.0, label="Max Grad Norm", info="0 disables clipping")
                                train_num_workers = gr.Number(value=4, label="Dataloader Workers", precision=0)
                            with gr.Row():
                                grad_accum = gr.Number(value=1, label="Grad Accumulation Steps", precision=0,
                                                      info="Effective batch = batch size × this. Use to simulate larger batches on limited VRAM")
                                compile_model = gr.Checkbox(label="torch.compile (experimental)", value=False,
                                                           info="Compiles the DiT transformer. Faster after warmup; best with triton installed")
                            freeze_adaln = gr.Checkbox(label="Freeze AdaLN and time embedding (on for single speaker, off for multi-speaker; auto-set)",
                                                      value=True)
                        with gr.Accordion("▶️ Resume Training", open=False):
                            resume_ckpt = gr.Dropdown(choices=list_resume_ckpts(), value='',
                                                     label="Resume from Checkpoint",
                                                     allow_custom_value=True,
                                                     info="Continue an interrupted run (restores optimizer, EMA and step count). "
                                                          "Leave empty to start fresh. Only full checkpoints are resumable")
                            refresh_resume_btn = gr.Button("🔄 Refresh Checkpoint List")
                            save_full_ckpt = gr.Checkbox(label="Save full checkpoints (resumable; larger files)", value=True,
                                                        info="Includes optimizer/EMA state so training can be resumed. "
                                                             "Uncheck to save weights-only checkpoints (smaller, not resumable)")
                        recipe_note = gr.Markdown("")
                        with gr.Row():
                            train_btn = gr.Button("🚀 Start Training", variant="primary")
                            stop_btn = gr.Button("🛑 Stop Training", variant="stop")
                        train_log = gr.Textbox(label="Training Log", lines=22, max_lines=22,
                                              interactive=False, autoscroll=True)
        
        def update_language(selected_language):
            global current_language
            current_language = selected_language
            lang = LANGUAGES[current_language]
            
            return [
                gr.update(label=lang["model_path_label"], placeholder=lang["model_path_placeholder"]),
                gr.update(value=lang["load_model_btn"]),
                gr.update(label=lang["input_audio_label"]),
                
                gr.update(label=lang["target_speaker"]),
                gr.update(label=lang["key_shift"]),
                gr.update(label=lang["infer_steps"], info=lang["infer_steps_info"]),
                gr.update(label=lang["use_fp16"], info=lang["use_fp16_info"]),
                gr.update(label=lang["pitch_filter"], info=lang["pitch_filter_info"]),
                gr.update(label=lang["batch_size"], info=lang["batch_size_info"]),
                
                gr.update(label=lang["basic_params"]),
                gr.update(label=lang["adv_cfg_params"]),
                gr.update(label=lang["slicer_params"]),
                
                gr.update(label=lang["ds_cfg_strength"], info=lang["ds_cfg_strength_info"]),
                gr.update(label=lang["spk_cfg_strength"], info=lang["spk_cfg_strength_info"]),
                gr.update(label=lang["skip_cfg_strength"], info=lang["skip_cfg_strength_info"]),
                gr.update(label=lang["cfg_skip_layers"], info=lang["cfg_skip_layers_info"]),
                gr.update(label=lang["cfg_rescale"], info=lang["cfg_rescale_info"]),
                gr.update(label=lang["cvec_downsample"], info=lang["cvec_downsample_info"]),
                
                gr.update(label=lang["slicer_threshold"], info=lang["slicer_threshold_info"]),
                gr.update(label=lang["slicer_min_length"], info=lang["slicer_min_length_info"]),
                gr.update(label=lang["slicer_min_interval"], info=lang["slicer_min_interval_info"]),
                gr.update(label=lang["slicer_hop_size"], info=lang["slicer_hop_size_info"]),
                gr.update(label=lang["slicer_max_sil"], info=lang["slicer_max_sil_info"]),
                
                gr.update(value=lang["convert_btn"]),
                gr.update(value=lang["output_section"]),
                gr.update(label=lang["output_audio_label"]),
                gr.update(value=lang["init_message"]),
                
                gr.update(value=f"""
                <div class="title">
                    <h1>{lang["main_title"]}</h1>
                </div>
                <div class="subtitle">
                    <h3>{lang["subtitle"]}</h3>
                </div>
                <div class="info-box">
                    <p>{lang["github_info"]}</p>
                </div>
                <div class="info-box">
                    <p>{lang["audio_note"]}</p>
                </div>
                """),
                
                gr.update(value=f"""
                <div class="info-box">
                    <h4>{lang["quick_tips"]}</h4>
                    {lang["tips_content"]}
                </div>
                """),

                gr.update(label=lang["sampling_params"]),
                gr.update(label=lang["ode_method"], info=lang["ode_method_info"]),
                gr.update(label=lang["sway_coef"], info=lang["sway_coef_info"]),
                gr.update(label=lang["seed"], info=lang["seed_info"]),
                gr.update(label=lang["tab_inference"]),
                gr.update(label=lang["tab_train"])
            ]
        
        def load_model_and_update_speakers(model_path):
            available_speakers, message = initialize_models(model_path)
            
            if available_speakers and len(available_speakers) > 0:
                return gr.update(choices=available_speakers, value=available_speakers[0]), message
            else:
                return gr.update(choices=[], value=None), message
        
        language_selector.change(
            fn=update_language,
            inputs=[language_selector],
            outputs=[
                model_path, reload_btn, input_audio,  
                speaker, key_shift, infer_steps, use_fp16, robust_f0, batch_size,  
                basic_params_accordion, adv_cfg_accordion, slicer_accordion,  
                ds_cfg_strength, spk_cfg_strength, skip_cfg_strength, 
                cfg_skip_layers, cfg_rescale, cvec_downsample_rate,
                slicer_threshold, slicer_min_length, slicer_min_interval,
                slicer_hop_size, slicer_max_sil_kept,
                convert_btn, output_markdown, output_audio, output_message,
                html_header, tips_html,
                sampling_accordion, ode_method, sway_coef, seed,
                inference_tab, train_tab
            ]
        )
        
        reload_btn.click(
            fn=load_model_and_update_speakers,
            inputs=[model_path],
            outputs=[speaker, output_message]
        )
        
        convert_btn.click(
            fn=lambda: LANGUAGES[current_language]["processing"],
            inputs=None,
            outputs=output_message,
            queue=False
        ).then(
            fn=process_with_progress,
            inputs=[
                input_audio, speaker, key_shift, infer_steps, robust_f0, use_fp16,
                batch_size,
                ds_cfg_strength, spk_cfg_strength, skip_cfg_strength, cfg_skip_layers, cfg_rescale, cvec_downsample_rate,
                slicer_threshold, slicer_min_length, slicer_min_interval, slicer_hop_size, slicer_max_sil_kept,
                ode_method, sway_coef, seed
            ],
            outputs=[output_audio, output_message],
            show_progress_on=output_audio
        )

        # --- Training tab events ---
        scan_btn.click(
            fn=scan_dataset,
            inputs=[dataset_dir],
            outputs=[dataset_summary]
        ).then(
            fn=suggest_speaker_settings,
            inputs=[train_data_dir, dataset_dir],
            outputs=[freeze_adaln, drop_spk_prob, recipe_note]
        )

        preprocess_btn.click(
            fn=run_preprocess,
            inputs=[dataset_dir, train_data_dir, preprocess_workers, preprocess_overwrite, num_test_per_speaker,
                    slice_enable, slice_threshold, slice_min_len_ms, slice_max_len_s],
            outputs=[preprocess_log]
        ).then(
            fn=suggest_speaker_settings,
            inputs=[train_data_dir, dataset_dir],
            outputs=[freeze_adaln, drop_spk_prob, recipe_note]
        )

        # Auto-detect the speaker recipe when the app opens (if data is already there)
        app.load(
            fn=suggest_speaker_settings,
            inputs=[train_data_dir, dataset_dir],
            outputs=[freeze_adaln, drop_spk_prob, recipe_note]
        )

        pretrained_ckpt.change(
            fn=guess_model_from_ckpt,
            inputs=[pretrained_ckpt],
            outputs=[model_name]
        )

        refresh_resume_btn.click(
            fn=lambda: gr.update(choices=list_resume_ckpts()),
            inputs=None,
            outputs=[resume_ckpt]
        )

        train_btn.click(
            fn=start_training,
            inputs=[
                train_data_dir, model_name, pretrained_ckpt, run_name,
                learning_rate, weight_decay, max_steps, warmup_ratio,
                train_batch_size, max_frame_len, save_per_steps, test_per_steps,
                drop_spk_prob, ema_decay, max_grad_norm,
                freeze_adaln, train_num_workers,
                grad_accum, compile_model,
                resume_ckpt, save_full_ckpt,
            ],
            outputs=[train_log]
        )

        stop_btn.click(
            fn=stop_training,
            inputs=None,
            outputs=[train_log]
        )

    return app

@click.command()
@click.option('--share', is_flag=True, help='Share the app')
@click.option('--language', default='En', help='Default language (en or 中文)')
def main(share=False, language='En'):
    global current_language
    if language in LANGUAGES:
        current_language = language
    else:
        current_language = 'En'
    
    app = create_ui()
    app.launch(share=share, quiet=True, prevent_thread_lock=True)
    print(f"RIFT-SVC GUI running at {app.local_url}")
    if share and getattr(app, 'share_url', None):
        print(f"Public link: {app.share_url}")
    app.block_thread()

if __name__ == "__main__":
    main()