# utils/ffmpeg_utils.py
import os
import subprocess
import torch # type: ignore
import threading # Only used if you re-implement run_subprocess_with_protection with threads
import logging
import tempfile 
from pathlib import Path 
import shutil # For shutil.which

# Relative import from the same package (utils)
from .terminal_utils import IS_WINDOWS

logger = logging.getLogger(__name__)
def create_video_from_image(image_path: Path, output_path: Path, duration: int = 8, zoom_direction: str = "in"):
    ffmpeg_path = get_ffmpeg_path()

    zoom_expr = "zoom+0.002" if zoom_direction == "in" else "zoom-0.002"
    zoom_expr = "if(lte(zoom,1.5),zoom+0.002,zoom)" if zoom_direction == "in" else "if(gte(zoom,1.0),zoom-0.002,zoom)"

    vf_filter = (
        f"scale=1920:1080,zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':d=240:s=1920x1080,"
        "format=yuv420p"
    )

    cmd = [
        ffmpeg_path,
        "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-vf", vf_filter,
        "-t", str(duration),
        "-r", "30",
        "-preset", "veryfast",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)

def get_ffmpeg_path():
    """Returns the path to ffmpeg executable."""
    local_ffmpeg_win = Path("./ffmpeg.exe")
    local_ffmpeg_unix = Path("./ffmpeg")

    if IS_WINDOWS and local_ffmpeg_win.exists() and local_ffmpeg_win.is_file():
        return str(local_ffmpeg_win.resolve())
    elif not IS_WINDOWS and local_ffmpeg_unix.exists() and local_ffmpeg_unix.is_file():
        return str(local_ffmpeg_unix.resolve())
    
    ffmpeg_in_path = shutil.which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path
    
    return "ffmpeg.exe" if IS_WINDOWS else "ffmpeg"


def get_ffprobe_path():
    """Returns the path to ffprobe executable."""
    local_ffprobe_win = Path("./ffprobe.exe")
    local_ffprobe_unix = Path("./ffprobe")

    if IS_WINDOWS and local_ffprobe_win.exists() and local_ffprobe_win.is_file():
        return str(local_ffprobe_win.resolve())
    elif not IS_WINDOWS and local_ffprobe_unix.exists() and local_ffprobe_unix.is_file():
        return str(local_ffprobe_unix.resolve())
        
    ffprobe_in_path = shutil.which("ffprobe")
    if ffprobe_in_path:
        return ffprobe_in_path
    return "ffprobe.exe" if IS_WINDOWS else "ffprobe"


def concatenate_audio_ffmpeg(audio_file_paths, output_path):
    """
    Concatenates multiple audio files into a single output file using FFmpeg.
    This version re-encodes the audio to ensure stream consistency.
    """
    ffmpeg_path = get_ffmpeg_path()
    if not audio_file_paths:
        logger.error("No audio files provided for concatenation.")
        return None

    list_file_content = ""
    for p in audio_file_paths:
        list_file_content += f"file '{Path(p).resolve().as_posix()}'\n"

    temp_list_file_path = None 
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as tmpf:
            tmpf.write(list_file_content)
            temp_list_file_path = tmpf.name
        logger.debug(f"Temporary file list for concatenation: {temp_list_file_path}")
        logger.debug(f"Content:\n{list_file_content}")

        output_extension = output_path.suffix.lower()
        audio_codec_options = []
        if output_extension == ".mp3":
            audio_codec_options.extend(["-c:a", "libmp3lame", "-q:a", "2"]) 
        elif output_extension in [".m4a", ".mp4", ".aac"]:
            audio_codec_options.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "1"]) 
        else: 
            logger.warning(f"Unknown output extension {output_extension} for concatenation. Attempting libmp3lame, ensure output name is .mp3 or adjust.")
            audio_codec_options.extend(["-c:a", "libmp3lame", "-q:a", "2"])


        concat_cmd = [
            ffmpeg_path,
            "-y", 
            "-f", "concat",
            "-safe", "0", 
            "-i", temp_list_file_path,
        ]
        concat_cmd.extend(audio_codec_options)
        concat_cmd.append(str(output_path.resolve()))
        
        logger.info(f"Concatenating ({'and re-encoding' if '-c:a' in audio_codec_options else 'by copying'}) {len(audio_file_paths)} audio files into {output_path}...")
        run_subprocess_with_protection(concat_cmd, "Concatenating audio files")
        
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Successfully concatenated audio to: {output_path}")
            return output_path
        else:
            logger.error(f"Concatenated audio file {output_path} was not created or is empty.")
            return None

    except Exception as e:
        logger.error(f"Error during audio concatenation: {e}", exc_info=True)
        return None
    finally:
        if temp_list_file_path and os.path.exists(temp_list_file_path):
            try:
                os.remove(temp_list_file_path)
                logger.debug(f"Removed temporary concatenation list file: {temp_list_file_path}")
            except OSError as e_remove:
                logger.warning(f"Could not remove temporary list file {temp_list_file_path}: {e_remove}")

def check_gpu_support_ffmpeg():
    """Check if GPU encoding for FFmpeg is supported and actually usable"""
    ffmpeg_path = get_ffmpeg_path()
    logger.info(f"Using ffmpeg path for GPU check: {ffmpeg_path}")

    try:
        # Ensure ffmpeg_path is not None and points to a file, otherwise log error and return "cpu"
        if not ffmpeg_path or not Path(ffmpeg_path).is_file():
            logger.error(f"FFmpeg executable not found or invalid path: '{ffmpeg_path}'. Defaulting to CPU.")
            return "cpu"
            
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True, timeout=15
        )
        encoder_output = result.stdout.lower() 

        # NVIDIA NVENC
        if "h264_nvenc" in encoder_output:
            logger.info("h264_nvenc encoder found. Testing usability...")
            # Increased resolution for NVENC test
            test_cmd_nvenc = [
                ffmpeg_path, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=1:d=1",
                "-c:v", "h264_nvenc", "-f", "null", "-",
            ]
            try:
                process = subprocess.run(test_cmd_nvenc, capture_output=True, text=True, check=True, timeout=10)
                logger.info("NVIDIA GPU encoding (h264_nvenc) is available and working!")
                return "nvidia"
            except subprocess.CalledProcessError as e:
                logger.warning(f"h264_nvenc test failed with exit code {e.returncode}. NVENC might not be usable. Stderr: {e.stderr.strip()}")
            except subprocess.TimeoutExpired:
                logger.warning(f"h264_nvenc test timed out. NVENC might not be usable.")
        
        # Intel QuickSync Video (QSV)
        if "h264_qsv" in encoder_output:
            logger.info("h264_qsv encoder found. Testing usability...")
            test_cmd_qsv = [ # Using a slightly larger resolution for QSV test as well, just in case.
                ffmpeg_path, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=1:d=1",
                "-c:v", "h264_qsv", "-f", "null", "-",
            ]
            try:
                process = subprocess.run(test_cmd_qsv, capture_output=True, text=True, check=True, timeout=10)
                logger.info("Intel QuickSync encoding (h264_qsv) is available and working!")
                return "intel"
            except subprocess.CalledProcessError as e:
                logger.warning(f"h264_qsv test failed with exit code {e.returncode}. QSV might not be usable. Stderr: {e.stderr.strip()}")
            except subprocess.TimeoutExpired:
                 logger.warning(f"h264_qsv test timed out. QSV might not be usable.")


        # VA-API (Common on Linux)
        if "h264_vaapi" in encoder_output and not IS_WINDOWS: 
            logger.info("h264_vaapi encoder found. Testing usability...")
            test_cmd_vaapi = [
                ffmpeg_path, "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=640x360:r=1:d=1", # Increased resolution
                "-vf", "format=nv12,hwupload", 
                "-c:v", "h264_vaapi", "-f", "null", "-",
            ]
            try:
                process = subprocess.run(test_cmd_vaapi, capture_output=True, text=True, check=True, timeout=10)
                logger.info("VA-API encoding (h264_vaapi) is available and working!")
                return "vaapi"
            except subprocess.CalledProcessError as e:
                logger.warning(f"h264_vaapi test failed with exit code {e.returncode}. VA-API might not be usable. Stderr: {e.stderr.strip()}")
            except subprocess.TimeoutExpired:
                logger.warning(f"h264_vaapi test timed out. VA-API might not be usable.")
        
        logger.info("No working FFmpeg hardware encoders (NVENC, QSV, VA-API for h264) positively identified or usable. Using CPU encoding.")
        return "cpu"
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg -encoders command timed out. Assuming no GPU support. Defaulting to CPU.")
        return "cpu"
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg -encoders command failed: {e.stderr.strip()}. Assuming no GPU support. Defaulting to CPU.")
        return "cpu"
    except FileNotFoundError:
        logger.error(f"FFmpeg executable not found at '{ffmpeg_path}'. Ensure FFmpeg is installed and in PATH or ./ . Defaulting to CPU.")
        return "cpu"
    except Exception as e:
        logger.error(f"Error checking FFmpeg GPU support: {e}. Defaulting to CPU encoding.", exc_info=True)
        return "cpu"

def check_pytorch_gpu_availability():
    """Check if GPU (CUDA, XPU, ROCm) is available for PyTorch."""
    try:
        if not torch: 
            logger.warning("PyTorch library is not available. Transcription will use CPU or fail if no alternative.")
            return False

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA is available for PyTorch! Found {device_count} device(s).")
            for i in range(device_count):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"    CUDA Capability: {torch.cuda.get_device_capability(i)}")
            return True
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            logger.info("Intel XPU (oneAPI) is available for PyTorch transcription.")
            return True
        elif hasattr(torch, "version") and hasattr(torch.version, "hip") and torch.version.hip and torch.cuda.is_available():
            is_rocm = False
            for i in range(torch.cuda.device_count()):
                if "amd" in torch.cuda.get_device_name(i).lower(): 
                    is_rocm = True
                    break
            if is_rocm:
                logger.info("AMD ROCm is available for PyTorch transcription (via CUDA interface).")
                return True
        
        logger.info("No GPU acceleration (CUDA, XPU, ROCm) available for PyTorch. Using CPU for transcription.")
        logger.info(f"PyTorch version: {torch.__version__}")
        return False
    except Exception as e:
        logger.error(f"Error checking PyTorch GPU availability: {e}", exc_info=True)
        return False

def run_subprocess_with_protection(cmd, desc="Running command"):
    logger.info(f"{desc}...")
    cmd_str = [str(c) for c in cmd] 
    logger.debug(f"Command: {' '.join(cmd_str)}") 

    process = None
    stdout_data_decoded = ""
    stderr_data_decoded = ""
    
    creation_flags = 0
    if IS_WINDOWS:
        creation_flags = 0x08000000 # subprocess.CREATE_NO_WINDOW

    try:
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
            text=True, 
            errors="ignore" 
        )
        
        stdout_data, stderr_data = process.communicate(timeout=3600) 

        stdout_data_decoded = stdout_data
        stderr_data_decoded = stderr_data

        if process.returncode != 0:
            logger.error(f"Error during '{desc}'. Return code: {process.returncode}")
            logger.error(f"Command: {' '.join(cmd_str)}")
            if stdout_data_decoded and stdout_data_decoded.strip(): logger.error(f"Stdout: {stdout_data_decoded.strip()}")
            if stderr_data_decoded and stderr_data_decoded.strip(): logger.error(f"Stderr: {stderr_data_decoded.strip()}")
            raise subprocess.CalledProcessError(
                process.returncode, cmd_str, output=stdout_data_decoded, stderr=stderr_data_decoded
            )
        else:
            logger.info(f"'{desc}' completed successfully.")
            if stdout_data_decoded and stdout_data_decoded.strip(): logger.debug(f"Stdout: {stdout_data_decoded.strip()}") 
            if stderr_data_decoded and stderr_data_decoded.strip(): logger.warning(f"Stderr (though successful): {stderr_data_decoded.strip()}")

        return stdout_data_decoded, stderr_data_decoded

    except subprocess.TimeoutExpired:
        logger.error(f"Command '{desc}' timed out after 1 hour.")
        if process:
            process.kill()
            stdout_data, stderr_data = process.communicate()
            stdout_data_decoded = stdout_data 
            stderr_data_decoded = stderr_data 
            if stdout_data_decoded and stdout_data_decoded.strip(): logger.error(f"Timeout Stdout: {stdout_data_decoded.strip()}")
            if stderr_data_decoded and stderr_data_decoded.strip(): logger.error(f"Timeout Stderr: {stderr_data_decoded.strip()}")
        raise 
    except Exception as e:
        logger.error(f"An unexpected error occurred in run_subprocess_with_protection for '{desc}': {e}", exc_info=True)
        raise