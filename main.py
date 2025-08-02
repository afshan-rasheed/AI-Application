# main.py
import argparse
import random
import sys
import re
import os
import atexit
from pathlib import Path
import logging
import yaml
import subprocess # Keep for CalledProcessError if needed elsewhere
import shutil
from collections import defaultdict

try:
    import torch
except ImportError:
    torch = None

try:
    import stable_whisper
except ImportError:
    stable_whisper = None

# Assuming utils are in a 'utils' subdirectory relative to main.py
# If your structure is different, you might need to adjust sys.path or use relative imports

try:
    from utils.terminal_utils import disable_quick_edit_mode, setup_signal_handlers
    from utils.subtitle_utils import STYLES
    from utils.ffmpeg_utils import concatenate_audio_ffmpeg
    from video_creator import create_youtube_video # video_creator.py should be in the same dir or Python path
    from video_creator import convert_images_to_videos
except ImportError as e:
    print(f"Error importing utility modules or video_creator: {e}")
    print("Please ensure 'video_creator.py' and the 'utils' directory are correctly placed and accessible.")
    sys.exit(1)


logger = logging.getLogger("VideoGenerator") # Will be configured by setup_logging

story_part_regex = re.compile(r"^(.*?)\s*(\d+)\s*\.\w+$")

AUDIO_EXTENSIONS_GLOB = ["*.mp3", "*.wav", "*.m4a", "*.ogg", "*.flac"]
IMAGE_EXTENSIONS_GLOB = ["*.jpg", "*.png", "*.jpeg", "*.webp"]
VIDEO_EXTENSIONS_GLOB = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]

IMAGE_EXTENSIONS_FOR_REGEX = [ext[2:] for ext in IMAGE_EXTENSIONS_GLOB]
VIDEO_EXTENSIONS_FOR_REGEX = [ext[2:] for ext in VIDEO_EXTENSIONS_GLOB]

IMAGE_DOT_SUFFIXES = [ext.replace('*', '') for ext in IMAGE_EXTENSIONS_GLOB]
VIDEO_DOT_SUFFIXES = [ext.replace('*', '') for ext in VIDEO_EXTENSIONS_GLOB]

DEFAULT_SETTINGS = {
    "gpu_acceleration": True,
    "transcription_model_size": "base",
    "subtitle_style": "Default",
    # "image_effect": "none", # Superseded by image_options
    "output_resolution": "1280x720",
    "initial_image_duration": 5,
    "audio_directory": "audio",
    "images_directory": "images",
    "output_directory": "output",
    "temp_directory": "temp_processing",
    "ffmpeg": {
        "video_quality": None, # Allow None, video_creator handles defaults
        "video_preset": None,  # Allow None
        "audio_bitrate": "192k",
        "extra_video_options": [],
        "extra_audio_options": []
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "log_file": "video_generator.log" # Default log file name
    },
    "video_effects": {
        "black_noise": {
            "enabled": False,
            "strength": 7,
            "temporal": True
        },
        "film_overlay": {
            "enabled": False,
            "file_path": None,
            "opacity": 0.15,
            "loop": True,
            "blend_mode": None
        }
    },
    # NEW: image_options for Rebound Swing and default effect
    "image_options": {
        "default_effect": "none",  # "kenburns" or "none"
        "rebound_swing": {
            "enabled": False,
            "apply_to_all_initial_images": True,
            "speed": 0.5,
            "max_zoom": 1.05,
            "pan_strength": 30
        }
    }
}

_temporary_files_to_cleanup = []

def cleanup_temporary_files():
    logger.info("Cleaning up temporary files...")
    temp_processing_dir_str = DEFAULT_SETTINGS.get("temp_directory", "temp_processing")
    # Cleanup explicitly tracked files
    for temp_file in list(_temporary_files_to_cleanup): # Iterate over a copy
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"Removed explicitly tracked temporary file: {temp_file}")
            if temp_file in _temporary_files_to_cleanup: # Check again
                 _temporary_files_to_cleanup.remove(temp_file)
        except OSError as e:
            logger.warning(f"Could not remove temporary file {temp_file}: {e}")
        except ValueError: pass # If already removed by another process or duplicate call

    # Cleanup leftover files in temp_processing_dir
    temp_processing_dir = Path(temp_processing_dir_str)
    if temp_processing_dir.exists() and temp_processing_dir.is_dir():
        logger.info(f"Checking for leftover files in temporary directory: {temp_processing_dir}")
        for item in temp_processing_dir.iterdir():
            try:
                if item.is_file():
                    logger.info(f"Removing leftover temporary file: {item}")
                    item.unlink()
                # Optionally, remove empty subdirectories if you create them
                # elif item.is_dir():
                #     shutil.rmtree(item) # Be careful with rmtree
            except OSError as e:
                logger.warning(f"Could not remove leftover temp file/dir {item} from {temp_processing_dir}: {e}")
atexit.register(cleanup_temporary_files)


def deep_update(source, overrides):
    """Recursively update a dict with overrides."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            deep_update(source[key], value)
        else:
            source[key] = value
    return source

def load_config(config_path_arg=None):
    # Create a deep copy of DEFAULT_SETTINGS
    settings = yaml.safe_load(yaml.safe_dump(DEFAULT_SETTINGS)) # Robust deep copy

    source_of_config_load_attempt = "script defaults"
    config_file_to_try = None

    if config_path_arg:
        config_file_to_try = Path(config_path_arg)
        print(f"INFO: --config specified. Attempting to load: {config_file_to_try}")
        source_of_config_load_attempt = f"specified file ('{config_path_arg}')"
    else:
        default_config_path = Path("config.yaml")
        if default_config_path.is_file():
            config_file_to_try = default_config_path
            print("INFO: No --config specified. Found and attempting to load default 'config.yaml'")
            source_of_config_load_attempt = "default 'config.yaml'"
        else:
            print("INFO: No --config specified and default 'config.yaml' not found. Using script defaults.")

    if config_file_to_try and config_file_to_try.is_file():
        try:
            with open(config_file_to_try, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            if user_config:
                settings = deep_update(settings, user_config)
                print(f"INFO: Successfully loaded and merged configuration from: {config_file_to_try}")
                source_of_config_load_attempt += " (loaded successfully)"
            else:
                print(f"INFO: Config file {config_file_to_try} was empty. Current settings based on defaults/previous merges.")
        except Exception as e:
            print(f"ERROR: Error loading or parsing config file {config_file_to_try}: {e}. Reverting to script defaults for safety.")
            settings = yaml.safe_load(yaml.safe_dump(DEFAULT_SETTINGS)) # Reset to defaults
            source_of_config_load_attempt = f"error loading {config_file_to_try}, reverted to script defaults"
    elif config_path_arg:
        print(f"WARNING: Specified config file '{config_path_arg}' not found. Using script defaults.")
        source_of_config_load_attempt = f"specified file ('{config_path_arg}' - not found), using script defaults"

    # Ensure temp_directory is always set for atexit cleanup
    if "temp_directory" not in settings or not settings["temp_directory"]:
        settings["temp_directory"] = DEFAULT_SETTINGS["temp_directory"]
    DEFAULT_SETTINGS["temp_directory"] = settings["temp_directory"] # Update global for atexit

    settings["_config_load_source_debug"] = source_of_config_load_attempt
    return settings

def setup_logging(log_config_dict):
    # Clear existing handlers from the root logger
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]: # Iterate over a copy
            root_logger.removeHandler(handler)
            handler.close() # Close handler before removing

    level_str = log_config_dict.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    fmt = log_config_dict.get("format", DEFAULT_SETTINGS["logging"]["format"])
    datefmt = log_config_dict.get("datefmt", DEFAULT_SETTINGS["logging"]["datefmt"])
    log_file = log_config_dict.get("log_file", DEFAULT_SETTINGS["logging"]["log_file"]) # Use default if not in config

    handlers = [logging.StreamHandler(sys.stdout)] # Always log to console
    if log_file and isinstance(log_file, str) and log_file.strip():
        try:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            handlers.append(file_handler)
            print(f"Logging to file: {log_file_path.resolve()}")
        except Exception as e:
            print(f"Error setting up file logging for '{log_file}': {e}. Logging to console only.")
    else:
        print(f"No log file specified or log_file is invalid. Defaulting to: {DEFAULT_SETTINGS['logging']['log_file']}. Logging to console only if that also fails.")
        # Attempt default log file if primary one failed or was null
        if not (log_file and isinstance(log_file, str) and log_file.strip()) and DEFAULT_SETTINGS["logging"]["log_file"]:
            try:
                default_log_path = Path(DEFAULT_SETTINGS["logging"]["log_file"])
                default_log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(default_log_path, mode='a', encoding='utf-8')
                handlers.append(file_handler)
                print(f"Using default log file: {default_log_path.resolve()}")
            except Exception as e:
                 print(f"Error setting up default file logging for '{DEFAULT_SETTINGS['logging']['log_file']}': {e}. Logging to console only.")


    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    global logger # Make sure the global logger instance is updated
    logger = logging.getLogger("VideoGenerator") # Get the named logger


def check_dependencies():
    all_ok = True
    print("\n--- Dependency Check ---")
    # FFmpeg
    ffmpeg_exe_name = "ffmpeg.exe" if os.name == 'nt' else "ffmpeg"
    ffprobe_exe_name = "ffprobe.exe" if os.name == 'nt' else "ffprobe"
    
    ffmpeg_path = shutil.which(ffmpeg_exe_name)
    local_ffmpeg = Path(f"./{ffmpeg_exe_name}")
    if ffmpeg_path: print(f"INFO: Found ffmpeg in PATH: {ffmpeg_path}")
    elif local_ffmpeg.exists() and local_ffmpeg.is_file(): print(f"INFO: Found ffmpeg locally: {local_ffmpeg.resolve()}")
    else: print("ERROR: ffmpeg not found in PATH or locally. FFmpeg is essential."); all_ok = False

    # FFprobe
    ffprobe_path = shutil.which(ffprobe_exe_name)
    local_ffprobe = Path(f"./{ffprobe_exe_name}")
    if ffprobe_path: print(f"INFO: Found ffprobe in PATH: {ffprobe_path}")
    elif local_ffprobe.exists() and local_ffprobe.is_file(): print(f"INFO: Found ffprobe locally: {local_ffprobe.resolve()}")
    else: print("ERROR: ffprobe not found in PATH or locally. ffprobe is essential."); all_ok = False

    # Python Libraries
    if yaml: print(f"INFO: PyYAML found. Version: {getattr(yaml, '__version__', 'Unknown')}")
    else: print("ERROR: PyYAML library not found. Run 'pip install PyYAML'."); all_ok = False

    if torch: print(f"INFO: PyTorch found. Version: {torch.__version__}")
    else: print("ERROR: PyTorch library not found. Run 'pip install torch torchvision torchaudio'. Needed for GPU transcription."); all_ok = False

    if stable_whisper: print("INFO: stable-whisper (stable-ts) found.")
    else: print("ERROR: stable-whisper library not found. Run 'pip install stable-ts'."); all_ok = False
    
    if not all_ok:
        print("\nERROR: Critical dependencies missing. Please install them and try again.")
        sys.exit(1)
    print("INFO: All checked dependencies seem to be present.")
    print("------------------------\n")
    return True


def discover_stories(audio_dir_path, visuals_dir_path):
    stories = defaultdict(lambda: {
        "audio_parts": [],
        "main_image_fallback": None,
        "initial_images_sequence": [],
        "single_background_video": None, # This could be looping or single-play based on context
        "video_sequence": [] # For numbered video sequences
    })

    logger.info(f"Scanning for audio files in: {audio_dir_path}")
    for ext_glob_pattern in AUDIO_EXTENSIONS_GLOB:
        for audio_file in audio_dir_path.glob(ext_glob_pattern):
            match = story_part_regex.match(audio_file.name)
            story_title_base = audio_file.stem # Default to stem if no regex match
            part_number = 1 # Default part number

            if match:
                story_title_base = match.group(1).strip()
                try:
                    part_number = int(match.group(2))
                except ValueError:
                    logger.warning(f"Could not parse part number from {audio_file.name}, treating as part 1 for base '{story_title_base}'")
                    part_number = 1 # Fallback
            else: # Handle cases where audio might not fit the regex but is still a part
                is_part_of_known_sequence = False
                for known_title in stories:
                    if story_title_base.startswith(known_title): # Basic check
                        # Attempt to extract number if possible, or assign a high number to sort later
                        num_match = re.search(r'(\d+)\s*$', story_title_base) # Number at the end
                        if num_match:
                            try: part_number = int(num_match.group(1))
                            except ValueError: pass
                        story_title_base = known_title # Assign to the known story
                        is_part_of_known_sequence = True
                        break
                if not is_part_of_known_sequence:
                     logger.debug(f"Audio file {audio_file.name} treated as single part for story '{story_title_base}'.")

            stories[story_title_base]["audio_parts"].append({"path": audio_file, "part_num": part_number})

    valid_story_titles = []
    for title in list(stories.keys()):
        if not stories[title]["audio_parts"]:
            del stories[title] # Remove stories with no audio
            continue
        # Sort audio parts by their part number
        stories[title]["audio_parts"].sort(key=lambda x: x["part_num"])
        stories[title]["audio_parts"] = [part["path"] for part in stories[title]["audio_parts"]] # Store only paths
        valid_story_titles.append(title)
        logger.info(f"Discovered story '{title}' with {len(stories[title]['audio_parts'])} audio part(s).")


    logger.info(f"Scanning for visual assets (images and videos) in: {visuals_dir_path}")
    for story_title in valid_story_titles:
        story_title_escaped = re.escape(story_title) # For regex matching

        # 1. Main Image Fallback (e.g., StoryTitle.png)
        for img_suffix in IMAGE_DOT_SUFFIXES:
            potential_image_name = f"{story_title}{img_suffix}"
            potential_image = visuals_dir_path / potential_image_name
            if potential_image.is_file():
                stories[story_title]["main_image_fallback"] = potential_image
                logger.info(f"Found main image fallback for story '{story_title}': {potential_image.name}")
                break # Found one, no need to check other extensions for this type

        # 2. Initial Images Sequence (e.g., StoryTitle image 01.png, StoryTitle img 02.jpg)
        initial_image_pattern = re.compile(
            rf"^{story_title_escaped}(?:\s(?:image|img|initial))?\s*(\d+)\.(?:{'|'.join(IMAGE_EXTENSIONS_FOR_REGEX)})$",
            re.IGNORECASE
        )
        temp_initial_images = []
        for item in visuals_dir_path.iterdir(): # Check all files in visuals_dir
            if item.is_file() and item.suffix.lower() in IMAGE_DOT_SUFFIXES:
                match = initial_image_pattern.match(item.name)
                if match:
                    try:
                        part_number = int(match.group(1)) # The number from the filename
                        temp_initial_images.append({"path": item, "num": part_number})
                    except ValueError:
                        logger.debug(f"Could not parse number from initial image candidate: {item.name}")
        if temp_initial_images:
            temp_initial_images.sort(key=lambda x: x["num"]) # Sort by number
            stories[story_title]["initial_images_sequence"] = [img_item["path"] for img_item in temp_initial_images]
            logger.info(f"Found {len(stories[story_title]['initial_images_sequence'])} initial images for story '{story_title}': {[p.name for p in stories[story_title]['initial_images_sequence'][:5]]}...")

        # 3. Numbered Video Sequence (e.g., StoryTitle video 01.mp4, StoryTitle vid 02.mov)
        video_sequence_pattern = re.compile(
            rf"^{story_title_escaped}(?:\s+(?:video|vid|sequence))?\s*(\d+)\.(?:{'|'.join(VIDEO_EXTENSIONS_FOR_REGEX)})$",
            re.IGNORECASE
        )
        temp_video_sequence = []
        for item in visuals_dir_path.iterdir():
            if item.is_file() and item.suffix.lower() in VIDEO_DOT_SUFFIXES:
                match = video_sequence_pattern.match(item.name)
                if match:
                    try:
                        part_number = int(match.group(1))
                        temp_video_sequence.append({"path": item, "num": part_number})
                    except ValueError:
                        logger.debug(f"Could not parse number from video sequence candidate: {item.name}")
        if temp_video_sequence:
            temp_video_sequence.sort(key=lambda x: x["num"])
            stories[story_title]["video_sequence"] = [vid_item["path"] for vid_item in temp_video_sequence]
            logger.info(f"Found {len(stories[story_title]['video_sequence'])} video sequence parts for story '{story_title}': {[p.name for p in stories[story_title]['video_sequence'][:5]]}...")

        # 4. Single Background Video (if no numbered video sequence was found) (e.g., StoryTitle.mp4)
        # This could be used as a looping background or a single-play intro depending on other assets.
        if not stories[story_title]["video_sequence"]: # Only if no numbered sequence
            for vid_suffix in VIDEO_DOT_SUFFIXES:
                potential_video_name = f"{story_title}{vid_suffix}"
                potential_video = visuals_dir_path / potential_video_name
                if potential_video.is_file():
                    stories[story_title]["single_background_video"] = potential_video
                    logger.info(f"Found single background/intro video for story '{story_title}': {potential_video.name}")
                    break
    return stories


def main():
    # Pre-parse for --config argument only, to load settings before full parsing
    temp_parser = argparse.ArgumentParser(add_help=False) # No help to avoid conflict
    temp_parser.add_argument("--config", help="Path to a YAML configuration file")
    pre_args, _ = temp_parser.parse_known_args() # Parse known args, ignore others for now

    settings = load_config(pre_args.config) # Load settings using the pre-parsed arg
    setup_logging(settings["logging"]) # Setup logging based on loaded settings

    logger.info(f"Initial config load attempt result: {settings.get('_config_load_source_debug', 'N/A')}")
    logger.info("Performing dependency checks...")
    check_dependencies()

    # Full argument parser
    parser = argparse.ArgumentParser(
        description="Create YouTube videos with audio, image(s)/video, and styled subtitles.",
        parents=[temp_parser], # Inherit --config
        conflict_handler='resolve' # Resolve conflicts if any (though --config is handled)
    )
    parser.add_argument("--concat-all", action="store_true", help="Concatenate all audio files in audio folder and loop all images to create one output video")
    parser.add_argument("--audio", help="Specific audio file to use (for single video mode, relative to audio_directory or absolute)")
    parser.add_argument("--image", help="Specific image file to use (main image, or first in sequence if --video also used, relative to images_directory or absolute)")
    parser.add_argument("--video", help="Specific background video file to use (for single video mode, relative to images_directory or absolute)")
    parser.add_argument("--output",help="Output video filename (for single video) or output directory (for story mode if it's a dir)")

    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--no-gpu", action="store_false", dest="gpu_acceleration_cli", help="Disable GPU acceleration (overrides config)")
    gpu_group.add_argument("--gpu", action="store_true", dest="gpu_acceleration_cli", help="Enable GPU acceleration (overrides config if it was false)")
    parser.set_defaults(gpu_acceleration_cli=None) # Default if neither is specified

    parser.add_argument("--process-stories", action="store_true", help="Process all discovered stories from audio/image directories")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size (overrides config)")
    parser.add_argument("--subtitle-style", choices=list(STYLES.keys()), help="Subtitle style (overrides config)")

    # CLI for image_options (default_effect) - Rebound Swing is only via config for now for simplicity
    parser.add_argument("--image-effect", choices=["none", "kenburns"],
                        help="Default image effect for main/last initial image (overrides image_options.default_effect in config)")

    parser.add_argument("--resolution", type=str, help="Output video resolution e.g., 1920x1080 (overrides config)")
    parser.add_argument("--initial-image-duration", type=int, help="Duration for each initial image (except the last one) in seconds (overrides config)")

    parser.add_argument("--video-quality", type=int, help="FFmpeg video quality (CRF/CQ, overrides config)")
    parser.add_argument("--video-preset", type=str, help="FFmpeg video preset (overrides config)")
    parser.add_argument("--audio-bitrate", type=str, help="FFmpeg audio bitrate (overrides config)")
    parser.add_argument("--extra-video-opts", nargs='+', help="Extra FFmpeg video options (replaces config list)")
    parser.add_argument("--extra-audio-opts", nargs='+', help="Extra FFmpeg audio options (replaces config list)")

    args = parser.parse_args()

    # --- Update settings based on CLI arguments ---
    # General settings
    if args.gpu_acceleration_cli is not None: use_gpu = args.gpu_acceleration_cli; gpu_source = "CLI"
    else: use_gpu = settings["gpu_acceleration"]; gpu_source = "Config/Default"

    model_size = args.model if args.model else settings["transcription_model_size"]; model_source = "CLI" if args.model else "Config/Default"
    subtitle_style_to_use = args.subtitle_style if args.subtitle_style else settings["subtitle_style"]; subtitle_source = "CLI" if args.subtitle_style else "Config/Default"
    output_resolution_to_use = args.resolution if args.resolution else settings["output_resolution"]; resolution_source = "CLI" if args.resolution else "Config/Default"
    initial_image_duration_sec = args.initial_image_duration if args.initial_image_duration is not None else settings.get("initial_image_duration", DEFAULT_SETTINGS["initial_image_duration"])
    initial_image_duration_source = "CLI" if args.initial_image_duration is not None else "Config/Default"

    # Image Options (handling CLI override for default_effect)
    image_options_final = settings.get("image_options", yaml.safe_load(yaml.safe_dump(DEFAULT_SETTINGS["image_options"]))) # Deep copy
    image_options_source = "Config/Default"
    if args.image_effect: # CLI --image-effect overrides image_options.default_effect
        image_options_final["default_effect"] = args.image_effect
        image_options_source = "CLI (for default_effect), rest from Config/Default"
    logger.info(f"Effective Image Options: {image_options_final} (Source: {image_options_source})")


    # Validate resolution
    try:
        if not isinstance(output_resolution_to_use, str) or 'x' not in output_resolution_to_use or len(output_resolution_to_use.split('x')) != 2:
            raise ValueError("Resolution must be a string in WxH format")
        w_str, h_str = output_resolution_to_use.split('x')
        if not (w_str.isdigit() and h_str.isdigit() and int(w_str) > 0 and int(h_str) > 0):
            raise ValueError("Width and Height must be positive integers.")
    except ValueError as e:
        logger.error(f"Invalid resolution: '{output_resolution_to_use}' (Source: {resolution_source}). Error: {e}. Using default {DEFAULT_SETTINGS['output_resolution']}.")
        output_resolution_to_use = DEFAULT_SETTINGS["output_resolution"]
        resolution_source = "Script Default (error)"

    # FFmpeg settings
    ffmpeg_settings_config = settings.get("ffmpeg", {})
    ffmpeg_settings_final = ffmpeg_settings_config.copy()
    ffmpeg_source_log = {k: "Config/Default" for k in DEFAULT_SETTINGS["ffmpeg"]}

    if args.video_quality is not None: ffmpeg_settings_final["video_quality"] = args.video_quality; ffmpeg_source_log["video_quality"] = "CLI"
    elif "video_quality" not in ffmpeg_settings_final: ffmpeg_settings_final["video_quality"] = DEFAULT_SETTINGS["ffmpeg"]["video_quality"]
    
    if args.video_preset: ffmpeg_settings_final["video_preset"] = args.video_preset; ffmpeg_source_log["video_preset"] = "CLI"
    elif "video_preset" not in ffmpeg_settings_final: ffmpeg_settings_final["video_preset"] = DEFAULT_SETTINGS["ffmpeg"]["video_preset"]

    if args.audio_bitrate: ffmpeg_settings_final["audio_bitrate"] = args.audio_bitrate; ffmpeg_source_log["audio_bitrate"] = "CLI"
    elif "audio_bitrate" not in ffmpeg_settings_final: ffmpeg_settings_final["audio_bitrate"] = DEFAULT_SETTINGS["ffmpeg"]["audio_bitrate"]

    if args.extra_video_opts: ffmpeg_settings_final["extra_video_options"] = args.extra_video_opts; ffmpeg_source_log["extra_video_options"] = "CLI"
    elif "extra_video_options" not in ffmpeg_settings_final: ffmpeg_settings_final["extra_video_options"] = DEFAULT_SETTINGS["ffmpeg"]["extra_video_options"]
    
    if args.extra_audio_opts: ffmpeg_settings_final["extra_audio_options"] = args.extra_audio_opts; ffmpeg_source_log["extra_audio_options"] = "CLI"
    elif "extra_audio_options" not in ffmpeg_settings_final: ffmpeg_settings_final["extra_audio_options"] = DEFAULT_SETTINGS["ffmpeg"]["extra_audio_options"]


    # Video Effects Settings
    video_effects_config = settings.get("video_effects", {})
    default_ve_config = DEFAULT_SETTINGS["video_effects"] # For easier access to defaults
    
    black_noise_specific_config = video_effects_config.get("black_noise", default_ve_config["black_noise"])
    apply_black_noise = black_noise_specific_config.get("enabled", default_ve_config["black_noise"]["enabled"])
    noise_strength = black_noise_specific_config.get("strength", default_ve_config["black_noise"]["strength"])
    noise_temporal = black_noise_specific_config.get("temporal", default_ve_config["black_noise"]["temporal"])
    
    film_overlay_specific_config = video_effects_config.get("film_overlay", default_ve_config["film_overlay"])
    apply_film_overlay = film_overlay_specific_config.get("enabled", default_ve_config["film_overlay"]["enabled"])
    film_overlay_file_str = film_overlay_specific_config.get("file_path", default_ve_config["film_overlay"]["file_path"])
    film_overlay_file_path = Path(film_overlay_file_str) if film_overlay_file_str else None
    if film_overlay_file_path and not film_overlay_file_path.is_absolute(): # Resolve if relative
        film_overlay_file_path = Path.cwd() / film_overlay_file_path # Relative to CWD
    
    film_overlay_opacity = float(film_overlay_specific_config.get("opacity", default_ve_config["film_overlay"]["opacity"]))
    film_overlay_loop = bool(film_overlay_specific_config.get("loop", default_ve_config["film_overlay"]["loop"]))
    film_overlay_blend_mode = film_overlay_specific_config.get("blend_mode", default_ve_config["film_overlay"]["blend_mode"])


    # Resolve directories
    project_root = Path.cwd() # Assuming script is run from project root
    audio_dir_path = project_root / settings["audio_directory"]
    visuals_dir_path = project_root / settings["images_directory"]
    output_root_dir_path = project_root / settings["output_directory"]
    temp_processing_dir = project_root / settings.get("temp_directory", DEFAULT_SETTINGS["temp_directory"])
    DEFAULT_SETTINGS["temp_directory"] = str(temp_processing_dir.resolve()) # For atexit

    # Setup terminal and signal handlers
    disable_quick_edit_mode()
    setup_signal_handlers()

    logger.info("--- YouTube Video Generator Initialized ---")
    logger.info(f"GPU Acceleration: {use_gpu} (Source: {gpu_source})")
    logger.info(f"Transcription Model: {model_size} (Source: {model_source})")
    logger.info(f"Subtitle Style: {subtitle_style_to_use} (Source: {subtitle_source})")
    logger.info(f"Output Resolution: {output_resolution_to_use} (Source: {resolution_source})")
    logger.info(f"Initial Image Duration (per image): {initial_image_duration_sec}s (Source: {initial_image_duration_source})")

    logger.info("FFmpeg Settings (effective):")
    for k_ffmpeg in DEFAULT_SETTINGS["ffmpeg"].keys():
        actual_val_ffmpeg = ffmpeg_settings_final.get(k_ffmpeg)
        source_msg_ffmpeg = ffmpeg_source_log.get(k_ffmpeg, "Config/Default" if k_ffmpeg in ffmpeg_settings_config else "Script Default")
        logger.info(f"  {k_ffmpeg.replace('_', ' ').title()}: {actual_val_ffmpeg if actual_val_ffmpeg is not None else 'Default/None'} (Source: {source_msg_ffmpeg})")

    logger.info("Video Effects Settings (effective):")
    logger.info(f"  Fine Grain Noise Enabled: {apply_black_noise} (Source: Config/Default)")
    if apply_black_noise: logger.info(f"    Strength: {noise_strength}, Temporal: {noise_temporal}")
    logger.info(f"  Film Overlay Enabled: {apply_film_overlay} (Source: Config/Default)")
    if apply_film_overlay:
        if film_overlay_file_path and film_overlay_file_path.exists():
            logger.info(f"    File: {film_overlay_file_path.resolve()}")
        else:
            logger.warning(f"    File: {film_overlay_file_path} (NOT FOUND - effect will be skipped if enabled)")
        logger.info(f"    Opacity: {film_overlay_opacity}, Loop: {film_overlay_loop}, Blend Mode: {film_overlay_blend_mode or 'alpha'}")
    
    logger.info(f"Audio Directory: {audio_dir_path.resolve()} (Source: Config/Default)")
    logger.info(f"Visuals Directory (Images & Videos): {visuals_dir_path.resolve()} (Source: Config/Default)")
    logger.info(f"Output Directory: {output_root_dir_path.resolve()} (Source: Config/Default)")
    logger.info(f"Temporary Processing Directory: {temp_processing_dir.resolve()} (Source: Config/Default)")
    logger.info("=" * 50)

    # Ensure directories exist
    for dir_path in [audio_dir_path, visuals_dir_path, output_root_dir_path, temp_processing_dir]:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}. Please check permissions.")
            sys.exit(1)


    jobs_to_process = []
    if args.concat_all:
        logger.info("Processing by concatenating all audio files and looping all images...")
        all_audio_files = []
        for ext in AUDIO_EXTENSIONS_GLOB:
            all_audio_files.extend(sorted(audio_dir_path.glob(ext)))
        if not all_audio_files:
            logger.error(f"No audio files found in {audio_dir_path}. Cannot proceed.")
            sys.exit(1)
        safe_output_name = "combined_video.mp4"
        concatenated_audio_path = temp_processing_dir / "combined_audio.mp3"
        combined_audio = concatenate_audio_ffmpeg([str(p) for p in all_audio_files], concatenated_audio_path)
        if not combined_audio:
            logger.error("Failed to concatenate all audio files. Exiting.")
            sys.exit(1)
        if str(combined_audio) not in _temporary_files_to_cleanup:
            _temporary_files_to_cleanup.append(str(combined_audio))

        # Prioritize video file "b 1.mp4" if exists, then add other images
        prioritized_video = visuals_dir_path / "b 1.mp4"
        all_images = []
        if prioritized_video.is_file():
            # Pass prioritized video as single_video_for_sequence
            single_video_for_sequence = prioritized_video
        else:
            single_video_for_sequence = None
        # Add other images except the prioritized video
        for f in visuals_dir_path.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_DOT_SUFFIXES and f != prioritized_video:
                all_images.append(f)

        if not all_images and not single_video_for_sequence:
            logger.error(f"No images or prioritized video found in {visuals_dir_path}. Cannot proceed.")
            sys.exit(1)

        output_video_path = output_root_dir_path / safe_output_name
        if args.output:
            output_path_arg = Path(args.output)
            if output_path_arg.is_dir():
                output_video_path = output_path_arg / safe_output_name
            else:
                output_video_path = output_path_arg
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        jobs_to_process.append({
            "audio": combined_audio,
            "main_image_file_fallback": None,
            "initial_images_list": all_images,
            "single_video_for_sequence": single_video_for_sequence,
            "looping_background_video": None,
            "video_sequence_files": [],
            "output": output_video_path,
            "is_temp_audio": True,
            "temp_audio_path": concatenated_audio_path,
            "idx": 1,
        })
    if args.process_stories:
        logger.info("Processing in story mode...")
        print("📸 Converting images to 7–8 sec videos with zoom effect...")
        # Removed hardcoded audio path that does not exist to prevent error
        # convert_images_to_videos("images", "output/clips", "audio/background.mp3")

        # New combined processing for all audio files and images as a single story
        all_audio_files = []
        for ext in AUDIO_EXTENSIONS_GLOB:
            all_audio_files.extend(sorted(audio_dir_path.glob(ext)))
        if not all_audio_files:
            logger.error(f"No audio files found in {audio_dir_path}. Cannot proceed.")
            sys.exit(1)
        safe_output_name = "combined_story_video.mp4"
        concatenated_audio_path = temp_processing_dir / "combined_story_audio.mp3"
        combined_audio = concatenate_audio_ffmpeg([str(p) for p in all_audio_files], concatenated_audio_path)
        if not combined_audio:
            logger.error("Failed to concatenate all audio files. Exiting.")
            sys.exit(1)
        if str(combined_audio) not in _temporary_files_to_cleanup:
            _temporary_files_to_cleanup.append(str(combined_audio))

        # Prioritize video file "b 1.mp4" if exists, then add other images
        prioritized_video = visuals_dir_path / "b 1.mp4"
        all_images = []
        if prioritized_video.is_file():
            # Pass prioritized video as single_video_for_sequence
            single_video_for_sequence = prioritized_video
        else:
            single_video_for_sequence = None
        # Add other images except the prioritized video
        for f in visuals_dir_path.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_DOT_SUFFIXES and f != prioritized_video:
                all_images.append(f)

        if not all_images and not single_video_for_sequence:
            logger.error(f"No images or prioritized video found in {visuals_dir_path}. Cannot proceed.")
            sys.exit(1)

        output_video_path = output_root_dir_path / safe_output_name
        if args.output:
            output_path_arg = Path(args.output)
            if output_path_arg.is_dir():
                output_video_path = output_path_arg / safe_output_name
            else:
                output_video_path = output_path_arg
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        jobs_to_process.append({
            "audio": combined_audio,
            "main_image_file_fallback": None,
            "initial_images_list": all_images,
            "single_video_for_sequence": single_video_for_sequence,
            "looping_background_video": None,
            "video_sequence_files": [],
            "output": output_video_path,
            "is_temp_audio": True,
            "temp_audio_path": concatenated_audio_path,
            "idx": 1,
            "title": "Combined Story"
        })
    else: # Single file mode
        logger.info("Processing in single file mode...")
        audio_f_path = None
        if args.audio:
            audio_f_path_cli = Path(args.audio)
            if audio_f_path_cli.is_file() and audio_f_path_cli.exists(): audio_f_path = audio_f_path_cli.resolve()
            else: # Try relative to audio_dir_path
                audio_f_path_rel = audio_dir_path / audio_f_path_cli.name
                if audio_f_path_rel.is_file() and audio_f_path_rel.exists(): audio_f_path = audio_f_path_rel.resolve()
            
            if not audio_f_path: logger.error(f"Audio '{args.audio}' not found (checked absolute and relative to audio_directory)."); sys.exit(1)
        else: logger.error("No --audio specified for single video mode."); sys.exit(1)

        cli_image_path = None
        if args.image:
            img_path_cli = Path(args.image)
            if img_path_cli.is_file() and img_path_cli.exists(): cli_image_path = img_path_cli.resolve()
            else: # Try relative to visuals_dir_path
                img_path_rel = visuals_dir_path / img_path_cli.name
                if img_path_rel.is_file() and img_path_rel.exists(): cli_image_path = img_path_rel.resolve()

            if not cli_image_path: logger.warning(f"CLI Image '{args.image}' not found. It will be ignored.")
        
        cli_video_path = None
        if args.video:
            vid_path_cli = Path(args.video)
            if vid_path_cli.is_file() and vid_path_cli.exists(): cli_video_path = vid_path_cli.resolve()
            else: # Try relative to visuals_dir_path
                vid_path_rel = visuals_dir_path / vid_path_cli.name
                if vid_path_rel.is_file() and vid_path_rel.exists(): cli_video_path = vid_path_rel.resolve()

            if not cli_video_path: logger.warning(f"CLI Video '{args.video}' not found. It will be ignored.")

        # Determine visual assets for single mode based on CLI inputs
        job_single_video_for_seq_single = None
        job_looping_bg_vid_single = None
        job_initial_imgs_single = []
        job_main_img_fb_single = None

        if cli_video_path: # If --video is provided
            if cli_image_path: # And --image is provided, video plays once, then image
                job_single_video_for_seq_single = cli_video_path
                job_initial_imgs_single = [cli_image_path]
            else: # Only --video, it becomes looping background
                job_looping_bg_vid_single = cli_video_path
        elif cli_image_path: # Only --image is provided
            job_initial_imgs_single = [cli_image_path] # Treated as an initial image sequence of one
        else: # No visuals from CLI, try "b 1.mp4" first, then random fallback from visuals_dir
            prioritized_video = visuals_dir_path / "b 1.mp4"
            if prioritized_video.is_file():
                job_single_video_for_seq_single = prioritized_video
                logger.info(f"No specific visual from CLI. Using prioritized video: {prioritized_video.name}")
                # Also look for images to use with the video
                all_image_files_single_mode = [f for f in visuals_dir_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_DOT_SUFFIXES and f != prioritized_video]
                if all_image_files_single_mode:
                    job_initial_imgs_single = all_image_files_single_mode
                    logger.info(f"Found {len(all_image_files_single_mode)} images to use with the video.")
            else:
                all_image_files_single_mode = [f for f in visuals_dir_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_DOT_SUFFIXES]
                if all_image_files_single_mode:
                    job_main_img_fb_single = random.choice(all_image_files_single_mode)
                    logger.info(f"No specific visual from CLI. Using random image from visuals directory: {job_main_img_fb_single.name}")
                else:
                     logger.error(f"No visual specified via CLI and no fallback images or 'b 1.mp4' found in '{visuals_dir_path}'. Cannot proceed for single mode.")
                     sys.exit(1)

        output_f_path = Path(args.output if args.output else output_root_dir_path / f"{audio_f_path.stem}_video.mp4")
        if output_f_path.is_dir(): # If user gave a directory for --output
            output_f_path = output_f_path / f"{audio_f_path.stem}_video.mp4"
        output_f_path.parent.mkdir(parents=True, exist_ok=True)

        jobs_to_process.append({
            "audio": audio_f_path,
            "main_image_file_fallback": job_main_img_fb_single,
            "initial_images_list": job_initial_imgs_single,
            "single_video_for_sequence": job_single_video_for_seq_single,
            "looping_background_video": job_looping_bg_vid_single,

            "video_sequence_files": [], # Not applicable in single file mode from CLI like this
            "output": output_f_path,
            "is_temp_audio": False,
            "temp_audio_path": None,
            "idx": 1,
            "title": audio_f_path.stem
        })


    total_jobs = len(jobs_to_process)
    if total_jobs == 0: logger.info("No video jobs to process."); return
    logger.info(f"Found {total_jobs} video(s) to process.")

    for job_info in jobs_to_process:
        logger.info(f"\n[{job_info['idx']}/{total_jobs}] Processing Video for: '{job_info['title']}'")
        logger.info(f"  Audio: {job_info['audio'].resolve()}")
        
        visual_assets_log = []
        if job_info.get('video_sequence_files'): visual_assets_log.append(f"Video Sequence: {[p.name for p in job_info['video_sequence_files']]}")
        if job_info.get('single_video_for_sequence'): visual_assets_log.append(f"Single Play Video: {job_info['single_video_for_sequence'].name}")
        if job_info.get('looping_background_video'): visual_assets_log.append(f"Looping BG Video: {job_info['looping_background_video'].name}")
        if job_info.get('initial_images_list'): visual_assets_log.append(f"Initial Images: {[p.name for p in job_info['initial_images_list']]}")
        if job_info.get('main_image_file_fallback'): visual_assets_log.append(f"Main Image Fallback: {job_info['main_image_file_fallback'].name}")
        if visual_assets_log: logger.info(f"  Visuals: {'; '.join(visual_assets_log)}")
        else: logger.info("  Visuals: None specified or found for this job (should not happen if logic is correct).")
        
        logger.info(f"  Output: {job_info['output'].resolve()}")

        # Ensure all paths passed to create_youtube_video are Path objects or None
        created_video_path_str = create_youtube_video(
            audio_file=Path(job_info["audio"]),
            main_image_file_fallback=Path(job_info["main_image_file_fallback"]) if job_info.get("main_image_file_fallback") else None,
            initial_images_list=[Path(p) for p in job_info.get("initial_images_list", [])],
            single_video_for_sequence=Path(job_info["single_video_for_sequence"]) if job_info.get("single_video_for_sequence") else None,
            looping_background_video=Path(job_info["looping_background_video"]) if job_info.get("looping_background_video") else None,
            video_sequence_files=[Path(p) for p in job_info.get("video_sequence_files", [])],
            output_file=Path(job_info["output"]),
            
            use_gpu=use_gpu,
            model_size=model_size,
            subtitle_style=subtitle_style_to_use,
            
            video_quality=ffmpeg_settings_final.get("video_quality"),
            video_preset=ffmpeg_settings_final.get("video_preset"),
            audio_bitrate=ffmpeg_settings_final.get("audio_bitrate"),
            extra_ffmpeg_video_options=ffmpeg_settings_final.get("extra_video_options"),
            extra_ffmpeg_audio_options=ffmpeg_settings_final.get("extra_audio_options"),
            
            # image_effect=image_effect_to_use, # Old parameter, now handled by image_options
            output_resolution=output_resolution_to_use,
            initial_image_duration_s=initial_image_duration_sec,
            
            apply_black_noise_effect=apply_black_noise,
            black_noise_strength=noise_strength,
            black_noise_temporal=noise_temporal,
            
            apply_film_overlay_effect=apply_film_overlay,
            film_overlay_file=film_overlay_file_path, # Already a Path object or None
            film_overlay_opacity=film_overlay_opacity,
            film_overlay_loop=film_overlay_loop,
            film_overlay_blend_mode=film_overlay_blend_mode,

            image_options=image_options_final # NEW: Pass the image_options dictionary
        )
        if created_video_path_str: logger.info(f"Successfully created video: {created_video_path_str}")
        else: logger.error(f"Failed to create video for: {job_info['title']}")

        # Cleanup temporary concatenated audio if it was created for this job
        if job_info["is_temp_audio"] and job_info["temp_audio_path"]:
            temp_audio_to_remove = str(job_info["temp_audio_path"]) # Path object to string
            if temp_audio_to_remove in _temporary_files_to_cleanup:
                try:
                    if os.path.exists(temp_audio_to_remove):
                        os.remove(temp_audio_to_remove)
                    _temporary_files_to_cleanup.remove(temp_audio_to_remove) # Remove from tracking list
                    logger.info(f"Cleaned temp audio (post-job): {temp_audio_to_remove}")
                except (OSError, ValueError) as e: # ValueError if already removed from list
                    logger.warning(f"Failed to clean temp audio {temp_audio_to_remove}: {e}")
            else:
                logger.debug(f"Temporary audio {temp_audio_to_remove} was not in cleanup list, might have been cleaned already or never added.")


        logger.info("-" * 50)

    logger.info("All processing finished.")
    # Explicit call to cleanup at the very end, though atexit should also handle it
    cleanup_temporary_files()


if __name__ == "__main__":
    main()
