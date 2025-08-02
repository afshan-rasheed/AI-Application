# video_creator.py
import os
import subprocess
try:
    import stable_whisper # type: ignore
except ImportError:
    stable_whisper = None
from pathlib import Path
import logging
import math
import sys # Added for sys.exit if utils are missing
try:
    import torch # Import torch to check for CUDA
except ImportError:
    torch = None

# Assuming utils are in a 'utils' subdirectory relative to video_creator.py

try:
    from utils.ffmpeg_utils import (
        get_ffmpeg_path,
        get_ffprobe_path,
        check_gpu_support_ffmpeg,
        # check_pytorch_gpu_availability, # Not strictly needed if transcription is forced to CPU
        run_subprocess_with_protection,

    )
    from utils.subtitle_utils import create_styled_ass
    from utils.ffmpeg_utils import create_video_from_image
    from utils.terminal_utils import IS_WINDOWS
except ImportError as e:
    print(f"CRITICAL ERROR in video_creator.py: Could not import utility modules: {e}")
    print("Please ensure 'utils' directory and its contents are accessible.")
    sys.exit(1)


logger = logging.getLogger(__name__)

NUMERIC_PI = 3.141592653589793



def convert_images_to_videos(image_dir, output_dir, audio_path):
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(Path(image_dir).glob("*.png")) + sorted(Path(image_dir).glob("*.jpg"))

    audio_duration = get_media_duration(Path(audio_path))
    if not audio_duration:
        print("⚠️ Unable to get audio duration. Defaulting to 8s per image.")
        clip_duration = 8
    else:
        clip_duration = max(5, audio_duration / len(image_files))  # minimum 5 seconds

    for idx, image_path in enumerate(image_files, start=1):
        output_path = Path(output_dir) / f"clip_{idx}.mp4"
        print(f"🎬 Creating video from: {image_path.name} → {output_path.name} [{clip_duration:.2f}s]")
        create_video_from_image(image_path, output_path, duration=int(clip_duration), zoom_direction="in")
        
def get_media_duration(media_file_path: Path) -> float | None:
    ffprobe_path = get_ffprobe_path()
    if not ffprobe_path or not Path(ffprobe_path).exists():
        logger.error(f"ffprobe executable not found at '{ffprobe_path}'. Cannot get media duration.")
        return None
    try:
        cmd = [
            ffprobe_path, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(media_file_path)
        ]
        duration_output = subprocess.check_output(cmd, text=True, errors="ignore").strip()

        if duration_output:
            return float(duration_output)
        else:
            logger.warning(f"ffprobe returned empty duration for {media_file_path}.")
            return None
    except subprocess.CalledProcessError as e:
        stderr_info = e.stderr.strip() if e.stderr and isinstance(e.stderr, str) else (e.stderr.decode(errors='ignore').strip() if e.stderr else 'N/A')
        logger.warning(f"ffprobe error getting duration for {media_file_path}. CMD: '{' '.join(e.cmd)}', RC: {e.returncode}, Stderr: {stderr_info}")
        return None
    except ValueError:
        logger.warning(f"Error parsing ffprobe duration output ('{duration_output if 'duration_output' in locals() else 'N/A'}') for {media_file_path}.")
        return None
    except FileNotFoundError:
        logger.error(f"ffprobe executable not found at '{ffprobe_path}' when trying to get duration for {media_file_path}.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting duration for {media_file_path}: {e}", exc_info=True)
        return None

def create_youtube_video(
    audio_file: Path,
    main_image_file_fallback: Path | None,
    initial_images_list: list[Path] | None,
    single_video_for_sequence: Path | None,
    looping_background_video: Path | None,
    video_sequence_files: list[Path] | None,
    output_file: Path,
    use_gpu=True, 
    model_size="base",
    subtitle_style="Default",
    video_quality=None, 
    video_preset=None,
    audio_bitrate="192k",
    extra_ffmpeg_video_options=None,
    extra_ffmpeg_audio_options=None,
    output_resolution="1280x720",
    initial_image_duration_s=5,
    apply_black_noise_effect: bool = False,
    black_noise_strength: int = 7,
    black_noise_temporal: bool = True,
    apply_film_overlay_effect: bool = False,
    film_overlay_file: Path | None = None,
    film_overlay_opacity: float = 0.15,
    film_overlay_loop: bool = True,
    film_overlay_blend_mode: str | None = None,
    image_options: dict | None = None
):
    if image_options is None:
        image_options = {
            "default_effect": "none",
            "rebound_swing": {
                "enabled": False, "apply_to_all_initial_images": False,
                "speed": 0.5, "max_zoom": 1.05, "pan_strength": 30,
            }
        }

    default_image_effect = image_options.get("default_effect", "none")
    rebound_swing_config = image_options.get("rebound_swing", {})
    rs_enabled = rebound_swing_config.get("enabled", False)
    # Use apply_to_all_initial_images from config, which was True in user's log
    rs_apply_to_all = rebound_swing_config.get("apply_to_all_initial_images", True)
    rs_speed = rebound_swing_config.get("speed", 0.09) # From user's log
    rs_max_zoom = rebound_swing_config.get("max_zoom", 1.002) # From user's log
    rs_pan_strength = rebound_swing_config.get("pan_strength", 21) # From user's log

    logger.info(f"Processing audio: {audio_file.name} with subtitle style: {subtitle_style}")
    logger.info(f"Default image effect (if Rebound Swing not active for an image): {default_image_effect}")
    if rs_enabled:
        logger.info(f"Rebound Swing effect ENABLED: apply_to_all={rs_apply_to_all}, speed={rs_speed}, max_zoom={rs_max_zoom}, pan_strength={rs_pan_strength}")
    else:
        logger.info("Rebound Swing effect DISABLED.")
    logger.info(f"Target output resolution: {output_resolution}")

    if apply_black_noise_effect: logger.info(f"Black noise effect ENABLED: strength={black_noise_strength}, temporal={black_noise_temporal}")
    else: logger.info("Black noise effect DISABLED.")

    if apply_film_overlay_effect and film_overlay_file and film_overlay_file.exists():
        logger.info(f"Film overlay effect ENABLED: file='{film_overlay_file.name}', opacity={film_overlay_opacity}, loop={film_overlay_loop}, blend='{film_overlay_blend_mode or 'default (alpha)'}'")
    elif apply_film_overlay_effect:
        logger.warning(f"Film overlay effect was enabled but file '{film_overlay_file}' not found or not specified. Overlay will be skipped.")
        apply_film_overlay_effect = False 
    else: logger.info("Film overlay effect DISABLED.")

    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- OPTIMIZED Transcription ---
    transcription_result = None
    use_fp16 = False
    if use_gpu and torch and torch.cuda.is_available():
        device_for_transcription = "cuda"
        use_fp16 = True # Enable FP16 for significant speedup on RTX cards
        logger.info("\nConfiguring transcription to use GPU (CUDA). FP16 enabled.")
    else:
        device_for_transcription = "cpu"
        logger.info("\nConfiguring transcription to use CPU.")

    try:
        logger.info(f"Loading '{model_size}' model on '{device_for_transcription}' for transcription...")
        if not stable_whisper:
            logger.error("stable_whisper not available. Cannot perform transcription.")
            return None
        model = stable_whisper.load_model(model_size, device=device_for_transcription)
        transcription_result = model.transcribe(str(audio_file), fp16=use_fp16) 
        logger.info("Transcription successful!")
    except Exception as e:
        logger.error(f"Transcription failed with '{model_size}' on '{device_for_transcription}': {e}", exc_info=True)
        return None

    if not transcription_result: 
        logger.error("Transcription result is empty. Cannot proceed.")
        return None

    subtitle_file_path = output_dir / f"{audio_file.stem}.ass"
    logger.info(f"\nGenerating '{subtitle_style}' subtitles: {subtitle_file_path}")
    create_styled_ass(str(subtitle_file_path), transcription_result.segments, subtitle_style, play_res_x_y=output_resolution)

    audio_duration = get_media_duration(audio_file)
    if audio_duration is None:
        logger.error(f"Could not determine audio duration for {audio_file}. Cannot proceed.")
        return None
    logger.info(f"Audio duration: {audio_duration:.3f} seconds")

    logger.info(f"\nCreating base video (without subtitles)...")
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path or not Path(ffmpeg_path).exists():
        logger.error(f"FFmpeg executable not found at '{ffmpeg_path}'. Cannot create video.")
        return None

    temp_video_path = output_dir / f"{output_file.stem}_temp_base.mp4"
    
    ffmpeg_gpu_type = check_gpu_support_ffmpeg() if use_gpu else "cpu"
    logger.info(f"FFmpeg GPU check determined type: {ffmpeg_gpu_type} (based on use_gpu: {use_gpu})")

    base_ffmpeg_cmd = [ffmpeg_path, "-y", "-nostdin"] 
    complex_filter_parts = [] 
    ffmpeg_visual_input_args = [] 
    current_ffmpeg_input_idx = 0 
    visual_stream_labels_for_concat = [] 
    accumulated_duration_for_images = 0.0

    try:
        target_w_str, target_h_str = output_resolution.split('x')
        target_w = int(target_w_str)
        target_h = int(target_h_str)
    except ValueError:
        logger.error(f"Invalid output_resolution '{output_resolution}'. Defaulting to 1280x720.")
        target_w, target_h = 1280, 720
    output_video_fps = 25 

    active_visual_source_type = "None"

    # Initialize final_visual_map_label - FIX for UnboundLocalError
    final_visual_map_label = ""

    # --- Full Visual Asset Processing Logic from your original script ---
    # This block is taken directly from your provided video_creator.py
    if video_sequence_files: 
        active_visual_source_type = "Video Sequence"
        logger.info(f"Processing numbered video sequence ({len(video_sequence_files)} videos).")
        for video_path in video_sequence_files:
            if not video_path.exists(): logger.warning(f"Video sequence file not found: {video_path}. Skipping."); continue
            ffmpeg_visual_input_args.extend(["-i", str(video_path)])
            video_input_label = f"[{current_ffmpeg_input_idx}:v]"
            filter_out_label = f"v_seq_{current_ffmpeg_input_idx}"
            complex_filter_parts.append(
                f"{video_input_label}scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p[{filter_out_label}]"
            )
            visual_stream_labels_for_concat.append(f"[{filter_out_label}]")
            vid_dur = get_media_duration(video_path)
            if vid_dur is not None: accumulated_duration_for_images += vid_dur
            else: logger.warning(f"Could not get duration for video sequence file {video_path.name}, this might affect timing.")
            current_ffmpeg_input_idx += 1
        if initial_images_list: logger.info(f"Initial images will play after the video sequence.")

    elif single_video_for_sequence and initial_images_list: 
        active_visual_source_type = "Single Video then Images with Loop"
        logger.info(f"Processing single video '{single_video_for_sequence.name}' to play first, followed by images, then loop video to fill remaining audio.")
        if not single_video_for_sequence.exists():
            logger.warning(f"Single video for sequence not found: {single_video_for_sequence}. Skipping this video.")
        else:
            # Add the single video input once
            ffmpeg_visual_input_args.extend(["-i", str(single_video_for_sequence)])
            video_input_label = f"[{current_ffmpeg_input_idx}:v]"
            filter_out_label = f"v_single_intro_{current_ffmpeg_input_idx}"
            complex_filter_parts.append(
                f"{video_input_label}scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p[{filter_out_label}]"
            )
            visual_stream_labels_for_concat.append(f"[{filter_out_label}]")
            vid_dur = get_media_duration(single_video_for_sequence)
            if vid_dur is not None: 
                accumulated_duration_for_images += vid_dur
                single_video_duration = vid_dur
            else: 
                logger.warning(f"Could not get duration for single intro video {single_video_for_sequence.name}.")
                single_video_duration = 5.0  # fallback duration
            current_ffmpeg_input_idx += 1

    elif single_video_for_sequence and not initial_images_list:
        active_visual_source_type = "Single Video Loop Only"
        logger.info(f"Processing single video '{single_video_for_sequence.name}' to loop for entire audio duration.")
        if not single_video_for_sequence.exists():
            logger.warning(f"Single video for sequence not found: {single_video_for_sequence}. Skipping this video.")
        else:
            vid_dur = get_media_duration(single_video_for_sequence)
            if vid_dur is not None: 
                single_video_duration = vid_dur
                loops_needed = math.ceil(audio_duration / single_video_duration)
                logger.info(f"Video duration: {single_video_duration:.3f}s, need {loops_needed} loop(s) to fill audio duration {audio_duration:.3f}s.")
            else: 
                logger.warning(f"Could not get duration for single video {single_video_for_sequence.name}.")
                single_video_duration = 5.0  # fallback duration
                loops_needed = math.ceil(audio_duration / single_video_duration)
            
            # Add stream_loop input for the video
            ffmpeg_visual_input_args.extend(["-stream_loop", str(loops_needed - 1), "-i", str(single_video_for_sequence)])
            video_input_label = f"[{current_ffmpeg_input_idx}:v]"
            filter_out_label = f"v_single_loop_only_{current_ffmpeg_input_idx}"
            complex_filter_parts.append(
                f"{video_input_label}scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p,"
                f"trim=duration={audio_duration:.3f}[{filter_out_label}]"
            )
            visual_stream_labels_for_concat.append(f"[{filter_out_label}]")
            current_ffmpeg_input_idx += 1

    if initial_images_list and (active_visual_source_type in ["None", "Video Sequence", "Single Video then Images", "Single Video then Images with Loop"] or not visual_stream_labels_for_concat):
        if active_visual_source_type == "None": active_visual_source_type = "Initial Images"
        logger.info(f"Processing {len(initial_images_list)} images " + ("as primary visuals." if not visual_stream_labels_for_concat else "after preceding videos."))
        num_images = len(initial_images_list)
        for i, img_path in enumerate(initial_images_list):
            if not img_path.exists(): logger.warning(f"Initial image file not found: {img_path}. Skipping."); continue
            ffmpeg_visual_input_args.extend(["-loop", "1", "-framerate", str(output_video_fps), "-i", str(img_path)])
            img_input_label = f"[{current_ffmpeg_input_idx}:v]"
            filter_out_label = f"v_img_{i}"
            is_last_image_in_list = (i == num_images - 1) and not video_sequence_files 
            
            current_segment_duration = 0.0
            # For "Single Video then Images with Loop", don't extend last image - use fixed duration for all images
            if is_last_image_in_list and active_visual_source_type != "Single Video then Images with Loop": 
                calculated_duration = audio_duration - accumulated_duration_for_images
                current_segment_duration = max(0.1, calculated_duration)
                logger.info(f"Last image '{img_path.name}' will play for calculated remaining {current_segment_duration:.3f}s.")
            else: 
                current_segment_duration = float(initial_image_duration_s)
                if active_visual_source_type != "Single Video then Images with Loop":
                    remaining_visuals_count = (num_images - (i + 1))
                    min_time_for_remaining = remaining_visuals_count * 0.1 
                    if accumulated_duration_for_images + current_segment_duration + min_time_for_remaining > audio_duration:
                         current_segment_duration = max(0.1, audio_duration - accumulated_duration_for_images - min_time_for_remaining)
                current_segment_duration = max(0.1, current_segment_duration)
                accumulated_duration_for_images += current_segment_duration
                logger.info(f"Image '{img_path.name}' (part {i+1}/{num_images}) will play for {current_segment_duration:.3f}s.")

            image_dynamic_effect_filter_str = ""
            total_frames_segment = max(1, int(current_segment_duration * output_video_fps))

            apply_rs_to_this_image = rs_enabled and (rs_apply_to_all or (is_last_image_in_list and not rs_apply_to_all))
            apply_kenburns_to_this_image = default_image_effect == "kenburns" and not apply_rs_to_this_image and (is_last_image_in_list or active_visual_source_type == "Initial Images")

            if apply_rs_to_this_image:
                logger.debug(f"Applying Rebound Swing to image {img_path.name} using 'on' variable.")
                frames_per_cycle_val = (output_video_fps / rs_speed) if rs_speed > 0 else total_frames_segment 
                if frames_per_cycle_val <= 0: frames_per_cycle_val = total_frames_segment 

                zoom_angle_factor = f"{NUMERIC_PI}/({frames_per_cycle_val})"
                pan_angle_factor = f"2*{NUMERIC_PI}/({frames_per_cycle_val})"

                zoom_expr = f"1+({rs_max_zoom}-1)*abs(sin(on*{zoom_angle_factor}))"
                pan_x_expr = f"(iw/2-iw/zoom/2)+{rs_pan_strength}*sin(on*{pan_angle_factor})"
                pan_y_expr = "(ih/2-ih/zoom/2)"
                image_dynamic_effect_filter_str = (
                    f",zoompan=z='{zoom_expr}':x='{pan_x_expr}':y='{pan_y_expr}':d={total_frames_segment}:s={target_w}x{target_h}:fps={output_video_fps}"
                )
            elif apply_kenburns_to_this_image and current_segment_duration > 0.01:
                logger.debug(f"Applying Ken Burns to image {img_path.name}")
                start_zoom_kb, end_zoom_kb = 1.0, 1.15 
                zoom_inc_kb = (end_zoom_kb - start_zoom_kb) / total_frames_segment if total_frames_segment > 1 else 0
                z_expr_kb = f"min({end_zoom_kb},max({start_zoom_kb},if(gte(on,0),{start_zoom_kb}+on*{zoom_inc_kb},{start_zoom_kb})))" 
                x_expr_kb = "(iw/2-(iw/zoom/2))"; y_expr_kb = "(ih/2-(ih/zoom/2))" 
                image_dynamic_effect_filter_str = (
                    f",zoompan=z='{z_expr_kb}':x='{x_expr_kb}':y='{y_expr_kb}':d={total_frames_segment}:s={target_w}x{target_h}:fps={output_video_fps}"
                )

            complex_filter_parts.append(
                f"{img_input_label}scale={target_w}:{target_h}:force_original_aspect_ratio=decrease," 
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1{image_dynamic_effect_filter_str},"
                f"format=yuv420p,trim=duration={current_segment_duration:.3f}[{filter_out_label}]"
            )
            visual_stream_labels_for_concat.append(f"[{filter_out_label}]")
            current_ffmpeg_input_idx += 1

    # Add looping single video after images for "Single Video then Images with Loop" mode
    if active_visual_source_type == "Single Video then Images with Loop" and single_video_for_sequence:
        # Calculate remaining time after images
        remaining_time_after_images = audio_duration - accumulated_duration_for_images
        if remaining_time_after_images > 0.1:  # Only add looping if significant time remains
            logger.info(f"Adding looping video '{single_video_for_sequence.name}' to fill remaining {remaining_time_after_images:.3f}s after images.")
            
            # Calculate how many loops we need
            loops_needed = math.ceil(remaining_time_after_images / single_video_duration)
            logger.info(f"Video duration: {single_video_duration:.3f}s, need {loops_needed} loop(s) to fill remaining time.")
            
            # Add stream_loop input for the same video
            ffmpeg_visual_input_args.extend(["-stream_loop", str(loops_needed - 1), "-i", str(single_video_for_sequence)])
            loop_video_input_label = f"[{current_ffmpeg_input_idx}:v]"
            loop_filter_out_label = f"v_single_loop_{current_ffmpeg_input_idx}"
            
            # Create filter for looped video with exact duration
            complex_filter_parts.append(
                f"{loop_video_input_label}scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p,"
                f"trim=duration={remaining_time_after_images:.3f}[{loop_filter_out_label}]"
            )
            visual_stream_labels_for_concat.append(f"[{loop_filter_out_label}]")
            current_ffmpeg_input_idx += 1
    
    base_ffmpeg_cmd.extend(ffmpeg_visual_input_args) 

    if visual_stream_labels_for_concat: 
        if len(visual_stream_labels_for_concat) > 1:
            concat_str = "".join(visual_stream_labels_for_concat)
            concat_output_label = "[visual_concat]"
            complex_filter_parts.append(f"{concat_str}concat=n={len(visual_stream_labels_for_concat)}:v=1:a=0{concat_output_label}")
            final_visual_map_label = concat_output_label
        else: 
            final_visual_map_label = visual_stream_labels_for_concat[0]
    elif looping_background_video: 
        if not looping_background_video.exists():
            logger.error(f"Looping background video not found: {looping_background_video}. Cannot proceed without visuals."); return None
        active_visual_source_type = "Looping Background Video"
        logger.info(f"Using single looping background video: {looping_background_video.name}")
        if not any(str(looping_background_video) in cmd_part for cmd_part in base_ffmpeg_cmd):
             base_ffmpeg_cmd.extend(["-stream_loop", "-1", "-i", str(looping_background_video)])
        
        bg_loop_output_label = "[bg_loop_scaled]"
        complex_filter_parts.append(
            f"[{current_ffmpeg_input_idx}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p{bg_loop_output_label}"
        )
        final_visual_map_label = bg_loop_output_label
        current_ffmpeg_input_idx += 1
    elif main_image_file_fallback: 
        if not main_image_file_fallback.exists():
            logger.error(f"Main fallback image not found: {main_image_file_fallback}. Cannot proceed without visuals."); return None
        active_visual_source_type = "Fallback Image"
        logger.info(f"Using single main image for entire duration: {main_image_file_fallback.name}")
        if not any(str(main_image_file_fallback) in cmd_part for cmd_part in base_ffmpeg_cmd):
             base_ffmpeg_cmd.extend(["-loop", "1", "-framerate", str(output_video_fps), "-i", str(main_image_file_fallback)])

        main_image_dynamic_effect_str = ""
        total_frames_main_img = max(1, int(audio_duration * output_video_fps))
        if rs_enabled: 
            logger.debug(f"Applying Rebound Swing to fallback image {main_image_file_fallback.name} using 'on' variable.")
            frames_per_cycle_val = (output_video_fps / rs_speed) if rs_speed > 0 else total_frames_main_img
            if frames_per_cycle_val <= 0: frames_per_cycle_val = total_frames_main_img
            zoom_angle_factor = f"{NUMERIC_PI}/({frames_per_cycle_val})"
            pan_angle_factor = f"2*{NUMERIC_PI}/({frames_per_cycle_val})"
            zoom_expr = f"1+({rs_max_zoom}-1)*abs(sin(on*{zoom_angle_factor}))"
            pan_x_expr = f"(iw/2-iw/zoom/2)+{rs_pan_strength}*sin(on*{pan_angle_factor})"
            pan_y_expr = "(ih/2-ih/zoom/2)"
            main_image_dynamic_effect_str = (f",zoompan=z='{zoom_expr}':x='{pan_x_expr}':y='{pan_y_expr}':d={total_frames_main_img}:s={target_w}x{target_h}:fps={output_video_fps}")
        elif default_image_effect == "kenburns":
            logger.debug(f"Applying Ken Burns to fallback image {main_image_file_fallback.name}")
            start_zoom_kb, end_zoom_kb = 1.0, 1.15
            zoom_inc_kb = (end_zoom_kb - start_zoom_kb) / total_frames_main_img if total_frames_main_img > 1 else 0
            z_expr_kb = f"min({end_zoom_kb},max({start_zoom_kb},if(gte(on,0),{start_zoom_kb}+on*{zoom_inc_kb},{start_zoom_kb})))"
            x_expr_kb = "(iw/2-(iw/zoom/2))"; y_expr_kb = "(ih/2-(ih/zoom/2))"
            main_image_dynamic_effect_str = (f",zoompan=z='{z_expr_kb}':x='{x_expr_kb}':y='{y_expr_kb}':d={total_frames_main_img}:s={target_w}x{target_h}:fps={output_video_fps}")

        main_img_output_label = "[main_img_scaled]"
        complex_filter_parts.append(
            f"[{current_ffmpeg_input_idx}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease," 
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1{main_image_dynamic_effect_str},"
            f"format=yuv420p{main_img_output_label}"
        )
        final_visual_map_label = main_img_output_label
        current_ffmpeg_input_idx += 1
    else: 
        if not final_visual_map_label: 
            logger.error("No visual source could be determined for FFmpeg. Cannot create video.")
            return None
    
    if apply_black_noise_effect and final_visual_map_label:
        noise_flags = "t+u" if black_noise_temporal else "p" 
        visual_after_fine_noise_label = final_visual_map_label.rstrip(']') + "_fine_noise]"
        noise_filter_stage = f"{final_visual_map_label}noise=c0s={black_noise_strength}:c0f={noise_flags}{visual_after_fine_noise_label}"
        complex_filter_parts.append(noise_filter_stage)
        final_visual_map_label = visual_after_fine_noise_label
        logger.info(f"Applied fine black noise. New visual stream: {final_visual_map_label}")

    if apply_film_overlay_effect and film_overlay_file and film_overlay_file.exists() and final_visual_map_label:
        overlay_input_opts = []
        is_video_overlay = film_overlay_file.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
        if film_overlay_loop and is_video_overlay: overlay_input_opts.extend(["-stream_loop", "-1"])
        elif not is_video_overlay: overlay_input_opts.extend(["-loop", "1", "-framerate", str(output_video_fps)])

        base_ffmpeg_cmd.extend(overlay_input_opts) 
        base_ffmpeg_cmd.extend(["-i", str(film_overlay_file)]) 
        
        overlay_input_stream_label = f"[{current_ffmpeg_input_idx}:v]" 
        visual_after_film_overlay_label = final_visual_map_label.rstrip(']') + "_film_overlayed]"
        processed_overlay_temp_label = "[overlay_processed]"
        main_stream_for_effect = final_visual_map_label 

        overlay_processing_filters = (
            f"{overlay_input_stream_label}scale=w={target_w}:h={target_h}:force_original_aspect_ratio=increase,"
            f"crop=w={target_w}:h={target_h}:x=(iw-ow)/2:y=(ih-oh)/2,setsar=1"
        )
        if film_overlay_blend_mode and film_overlay_blend_mode not in ['default (alpha)', 'alpha']: 
            main_stream_ready_for_blend = main_stream_for_effect.rstrip(']') + "_yuv420p_forblend]"
            complex_filter_parts.append(f"{main_stream_for_effect}format=yuv420p{main_stream_ready_for_blend}")
            overlay_processing_filters += f",format=yuv420p{processed_overlay_temp_label}"
            complex_filter_parts.append(overlay_processing_filters)
            blend_filter_command = (f"{main_stream_ready_for_blend}{processed_overlay_temp_label}blend=all_mode={film_overlay_blend_mode}:all_opacity={film_overlay_opacity}{visual_after_film_overlay_label}")
            complex_filter_parts.append(blend_filter_command)
            logger.info(f"Applying blend mode '{film_overlay_blend_mode}' for film overlay.")
        else: 
            overlay_processing_filters += f",format=rgba,colorchannelmixer=aa={film_overlay_opacity}{processed_overlay_temp_label}"
            complex_filter_parts.append(overlay_processing_filters)
            main_stream_rgba_for_overlay_label = main_stream_for_effect.rstrip(']') + "_rgba_for_overlay]"
            complex_filter_parts.append(f"{main_stream_for_effect}format=rgba{main_stream_rgba_for_overlay_label}")
            overlay_filter_command = (f"{main_stream_rgba_for_overlay_label}{processed_overlay_temp_label}overlay=x=0:y=0:shortest=0{visual_after_film_overlay_label}")
            complex_filter_parts.append(overlay_filter_command)
            logger.info(f"Applying alpha overlay method (RGBA base and overlay, opacity: {film_overlay_opacity}).")
        final_visual_map_label = visual_after_film_overlay_label
        current_ffmpeg_input_idx += 1 
        logger.info(f"Applied film overlay. New visual stream: {final_visual_map_label}")
    # --- End of Full Visual Asset Processing ---

    audio_input_actual_index = current_ffmpeg_input_idx 
    if not any(str(audio_file) in cmd_part for cmd_part in base_ffmpeg_cmd):
        base_ffmpeg_cmd.extend(["-i", str(audio_file)])
    audio_direct_map_specifier = f"{audio_input_actual_index}:a" 
    logger.debug(f"Audio input '{audio_file.name}' conceptually at FFmpeg input index {audio_input_actual_index}, map specifier: {audio_direct_map_specifier}")

    if complex_filter_parts:
        base_ffmpeg_cmd.extend(["-filter_complex", ";".join(complex_filter_parts)])

    if not final_visual_map_label: 
        logger.error("Critical: Final visual map label is empty after all visual processing. Cannot build FFmpeg command.")
        return None
    base_ffmpeg_cmd.extend(["-map", final_visual_map_label, "-map", audio_direct_map_specifier])

    # --- MODIFIED BASE VIDEO ENCODING OPTIONS ---
    video_encoding_opts = []
    logger.info(f"Base video FFmpeg GPU type: {ffmpeg_gpu_type}")
    # Use video_preset and video_quality from config/args
    cfg_video_preset = video_preset if video_preset else None # e.g. "fast"
    cfg_video_quality = str(video_quality if video_quality is not None else "19") # e.g. "19"

    current_extra_video_opts_list = list(extra_ffmpeg_video_options or [])

    if ffmpeg_gpu_type == "nvidia":
        nvenc_preset = cfg_video_preset if cfg_video_preset else "p5" # Default to p5 if config preset is None
        logger.info(f"Using NVIDIA NVENC (GPU) for base video. Preset: {nvenc_preset}, CQ: {cfg_video_quality}")
        video_encoding_opts.extend([
            "-c:v", "h264_nvenc", "-preset", nvenc_preset, "-tune", "hq",
            "-rc", "vbr", "-cq", cfg_video_quality, "-b:v", "0", "-profile:v", "high"
        ])
    elif ffmpeg_gpu_type == "intel":
        qsv_preset = cfg_video_preset if cfg_video_preset else "medium" # Default to medium if config preset is None
        logger.info(f"Using Intel QSV (GPU) for base video. Preset: {qsv_preset}, Global Quality: {cfg_video_quality}")
        video_encoding_opts.extend([
            "-c:v", "h264_qsv", "-preset", qsv_preset, "-global_quality", cfg_video_quality,
            "-b:v", "5M", "-low_power", "0", "-threads", "8"
        ])
    elif ffmpeg_gpu_type == "vaapi": 
        vaapi_preset = cfg_video_preset if cfg_video_preset else "medium"
        vaapi_quality = cfg_video_quality if video_quality is not None else "23" # QP for VAAPI
        logger.info(f"Using VA-API (GPU) for base video. Preset: {vaapi_preset}, QP: {vaapi_quality}")
        video_encoding_opts.extend(["-c:v", "h264_vaapi", "-preset", vaapi_preset, "-qp", vaapi_quality])
    else: # CPU
        cpu_preset = cfg_video_preset if cfg_video_preset else "medium"
        cpu_quality = cfg_video_quality if video_quality is not None else "23" # CRF for libx264
        logger.info(f"Using libx264 (CPU) for base video. Preset: {cpu_preset}, CRF: {cpu_quality}")
        video_encoding_opts.extend(["-c:v", "libx264", "-preset", cpu_preset, "-crf", cpu_quality])
        has_tune_in_extra = any("-tune" in str(opt).lower() for opt in current_extra_video_opts_list)
        if not has_tune_in_extra:
            is_static_image_dominant = (active_visual_source_type == "Fallback Image" and not rs_enabled and default_image_effect == "none" and not apply_black_noise_effect and not apply_film_overlay_effect)
            video_encoding_opts.extend(["-tune", "stillimage" if is_static_image_dominant else "film"])

    if current_extra_video_opts_list: 
        video_encoding_opts.extend(current_extra_video_opts_list)
    base_ffmpeg_cmd.extend(video_encoding_opts)
    # --- END OF MODIFIED BASE VIDEO ENCODING OPTIONS ---

    audio_opts = ["-c:a", "aac", "-b:a", audio_bitrate]
    if extra_ffmpeg_audio_options: audio_opts.extend(list(extra_ffmpeg_audio_options))
    base_ffmpeg_cmd.extend(audio_opts)

    # Common FFmpeg flags from original script
    base_ffmpeg_cmd.extend(["-r", str(output_video_fps)]) 
    if not (ffmpeg_gpu_type == "nvidia" or ffmpeg_gpu_type == "intel"): # If not using the new GPU params that include -threads
         base_ffmpeg_cmd.extend(["-threads", "0"]) # Original script used 0 for auto unless set by GPU flags
    base_ffmpeg_cmd.extend(["-pix_fmt", "yuv420p"])

    if active_visual_source_type == "Looping Background Video" or video_sequence_files or len(visual_stream_labels_for_concat) > 1:
        base_ffmpeg_cmd.append("-shortest")
    base_ffmpeg_cmd.extend(["-t", f"{audio_duration:.3f}"])
    base_ffmpeg_cmd.append(str(temp_video_path))

    try:
        run_subprocess_with_protection(base_ffmpeg_cmd, "Creating base video")
        logger.info(f"Base video created: {temp_video_path}")

        logger.info("\nAdding subtitles and finalizing video...")
        subtitle_path_for_ffmpeg = str(subtitle_file_path).replace("\\", "/")
        if IS_WINDOWS and ':' in subtitle_path_for_ffmpeg and subtitle_path_for_ffmpeg[1] == ':':
            subtitle_path_for_ffmpeg = subtitle_path_for_ffmpeg[0] + '\\:' + subtitle_path_for_ffmpeg[2:]
        
        final_video_filters_sub = [f"ass=filename='{subtitle_path_for_ffmpeg}'"]
        logger.debug(f"Subtitle filter string: {final_video_filters_sub[0]}")

        # --- MODIFIED SUBTITLE ENCODING OPTIONS ---
        final_video_encoding_opts_sub = []
        logger.info(f"Subtitle encoding FFmpeg GPU type: {ffmpeg_gpu_type}")

        if ffmpeg_gpu_type == "nvidia":
            nvenc_preset_sub = cfg_video_preset if cfg_video_preset else "p5"
            logger.info(f"Using NVIDIA NVENC (GPU) for subtitle encoding. Preset: {nvenc_preset_sub}, CQ: {cfg_video_quality}")
            final_video_encoding_opts_sub.extend([
                "-c:v", "h264_nvenc", "-preset", nvenc_preset_sub, "-tune", "hq", "-rc", "vbr",
                "-cq", cfg_video_quality, "-b:v", "0", "-profile:v", "high"
            ])
        elif ffmpeg_gpu_type == "intel":
            qsv_preset_sub = cfg_video_preset if cfg_video_preset else "medium"
            logger.info(f"Using Intel QSV (GPU) for subtitle encoding. Preset: {qsv_preset_sub}, Global Quality: {cfg_video_quality}")
            final_video_encoding_opts_sub.extend([
                "-c:v", "h264_qsv", "-preset", qsv_preset_sub, "-global_quality", cfg_video_quality,
                "-b:v", "5M", "-low_power", "0", "-threads", "8"
            ])
        elif ffmpeg_gpu_type == "vaapi": 
            vaapi_preset_sub = cfg_video_preset if cfg_video_preset else "medium"
            vaapi_quality_sub = cfg_video_quality if video_quality is not None else "23"
            logger.info(f"Using VA-API (GPU) for subtitle encoding. Preset: {vaapi_preset_sub}, QP: {vaapi_quality_sub}")
            final_video_encoding_opts_sub.extend(["-c:v", "h264_vaapi", "-preset", vaapi_preset_sub, "-qp", vaapi_quality_sub])
        else: # CPU
            cpu_preset_sub = cfg_video_preset if cfg_video_preset else "medium"
            cpu_quality_sub = cfg_video_quality if video_quality is not None else "23"
            logger.info(f"Using libx264 (CPU) for subtitle encoding. Preset: {cpu_preset_sub}, CRF: {cpu_quality_sub}")
            final_video_encoding_opts_sub.extend(["-c:v", "libx264", "-preset", cpu_preset_sub, "-crf", cpu_quality_sub])
            has_tune_in_extra_sub = any("-tune" in str(opt).lower() for opt in current_extra_video_opts_list)
            if not has_tune_in_extra_sub:
                 is_static_image_dominant_sub = (active_visual_source_type == "Fallback Image" and not rs_enabled and default_image_effect == "none" and not apply_black_noise_effect and not apply_film_overlay_effect)
                 final_video_encoding_opts_sub.extend(["-tune", "stillimage" if is_static_image_dominant_sub else "film"])

        if current_extra_video_opts_list: 
             final_video_encoding_opts_sub.extend(current_extra_video_opts_list)
        # --- END OF MODIFIED SUBTITLE ENCODING OPTIONS ---

        subtitle_ffmpeg_cmd = [
            ffmpeg_path, "-y", "-nostdin", "-i", str(temp_video_path),
            "-vf", ",".join(final_video_filters_sub)
        ]
        subtitle_ffmpeg_cmd.extend(final_video_encoding_opts_sub)
        subtitle_ffmpeg_cmd.extend(["-c:a", "copy"]) 
        subtitle_ffmpeg_cmd.extend(["-pix_fmt", "yuv420p", "-r", str(output_video_fps)])
        subtitle_ffmpeg_cmd.append(str(output_file))

        run_subprocess_with_protection(subtitle_ffmpeg_cmd, "Adding subtitles and finalizing video")
        logger.info(f"\nVideo with subtitles created: {output_file}")

        if temp_video_path.exists():
            try: os.remove(temp_video_path); logger.debug(f"Temp base video {temp_video_path} removed.")
            except OSError as e_rem: logger.warning(f"Could not remove temp base video {temp_video_path}: {e_rem}")
        return str(output_file)

    # Restoring full error handling from your original script
    except subprocess.CalledProcessError as e:
        error_desc = "unknown FFmpeg step"
        # More detailed error description logic from original script
        if hasattr(e, 'desc') and e.desc: 
            error_desc = e.desc
        elif isinstance(e, subprocess.CalledProcessError) and hasattr(e, 'cmd') and e.cmd:
             cmd_str_for_check = ' '.join(map(str, e.cmd))
             if temp_video_path.name in cmd_str_for_check: error_desc = "Creating base video"
             elif subtitle_file_path.name in cmd_str_for_check : error_desc = "Adding subtitles"

        logger.error(f"FFmpeg video processing error during '{error_desc}'.")
        cmd_str_list = [str(part) for part in e.cmd] if hasattr(e, 'cmd') and e.cmd is not None else []
        if cmd_str_list: logger.error(f"Failed FFmpeg command: {' '.join(cmd_str_list)}")
        
        stderr_output = ""
        if hasattr(e, 'stderr') and e.stderr:
            try: stderr_output = e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors='ignore')
            except Exception: pass 
            if stderr_output.strip(): logger.error(f"FFmpeg stderr:\n{stderr_output.strip()}")
        
        if temp_video_path.exists() and temp_video_path.stat().st_size > 0 and not output_file.exists():
             try:
                incomplete_name = output_file.with_name(output_file.stem + "_INCOMPLETE_BASE.mp4")
                if incomplete_name.exists(): incomplete_name.unlink(missing_ok=True)
                os.rename(temp_video_path, incomplete_name)
                logger.warning(f"FFmpeg process failed. Base video (no subtitles) saved as: {incomplete_name}")
             except Exception as rename_error: logger.error(f"Error renaming temporary file: {rename_error}", exc_info=True)
        return None
    except Exception as e_gen:
        logger.error(f"Unexpected error during video creation: {e_gen}", exc_info=True)
        if temp_video_path.exists() and temp_video_path.stat().st_size > 0 and not output_file.exists():
            try:
                unexpected_err_name = output_file.with_name(output_file.stem + "_INCOMPLETE_UNEXPECTED_ERR.mp4")
                if unexpected_err_name.exists(): unexpected_err_name.unlink(missing_ok=True)
                os.rename(temp_video_path, unexpected_err_name)
                logger.warning(f"Process failed unexpectedly. Base video (no subtitles) saved as: {unexpected_err_name}")
            except Exception as rename_error: logger.error(f"Error renaming temporary file: {rename_error}", exc_info=True)
        return None