import os
import uuid
import shutil
import torch
import requests
import subprocess
import librosa
import soundfile as sf
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from transformers import WhisperModel
import uvicorn

from realtime_inference import Avatar, load_config, load_all_model, FaceParsing, AudioProcessor
import realtime_inference  # Needed to patch global variables

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins (open to everyone)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
avatar_cache = {}
# Directory to store uploaded files
UPLOAD_FOLDER = "tmp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def validate_and_convert_audio(input_path: str, output_path: str) -> bool:
    """
    Validate and convert audio file to a compatible format.
    Returns True if successful, False otherwise.
    """
    try:
        # Check if file exists and has content
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            print(f"Audio file does not exist or is empty: {input_path}")
            return False
        
        # Try to load audio with librosa
        try:
            audio_data, sample_rate = librosa.load(input_path, sr=16000)  # Whisper expects 16kHz
            if len(audio_data) == 0:
                print("Audio file contains no audio data")
                return False
        except Exception as e:
            print(f"Failed to load audio with librosa: {e}")
            # Try with soundfile as fallback
            try:
                audio_data, sample_rate = sf.read(input_path)
                # Resample if needed
                if sample_rate != 16000:
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
            except Exception as e2:
                print(f"Failed to load audio with soundfile: {e2}")
                return False
        
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = librosa.to_mono(audio_data)
        
        # Save the processed audio
        sf.write(output_path, audio_data, sample_rate, format='WAV')
        print(f"Audio successfully processed and saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return False

def convert_audio_with_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    Convert audio using FFmpeg as a fallback method.
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",      # Mono
            "-f", "wav",     # WAV format
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Audio converted successfully with FFmpeg: {output_path}")
            return True
        else:
            print(f"FFmpeg conversion failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"FFmpeg conversion error: {e}")
        return False

# Load configuration and models once
args = load_config()
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

vae, unet, pe = load_all_model(
    unet_model_path=args.unet_model_path,
    vae_type=args.vae_type,
    unet_config=args.unet_config,
    device=device
)
timesteps = torch.tensor([0], device=device)

pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)

# Initialize audio processor and whisper model
audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
weight_dtype = unet.model.dtype
whisper = WhisperModel.from_pretrained(args.whisper_dir).to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

# Initialize face parser
fp = FaceParsing(
    left_cheek_width=args.left_cheek_width,
    right_cheek_width=args.right_cheek_width
) if args.version == "v15" else FaceParsing()

# Patch required global objects into realtime_inference module
realtime_inference.args = args
realtime_inference.device = device
realtime_inference.vae = vae
realtime_inference.unet = unet
realtime_inference.pe = pe
realtime_inference.timesteps = timesteps
realtime_inference.audio_processor = audio_processor
realtime_inference.whisper = whisper
realtime_inference.fp = fp
realtime_inference.weight_dtype = weight_dtype

# Inference wrapper
def run_inference(avatar_id: str, audio_path: str) -> str:
    if avatar_id not in avatar_cache:
        video_path = f"data/avatars/{avatar_id}"
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=args.bbox_shift,
            batch_size=args.batch_size,
            preparation=False
        )
        avatar_cache[avatar_id] = avatar
    else:
        avatar = avatar_cache[avatar_id]

    output_name = "api_output"
    avatar.inference(
        audio_path=audio_path,
        out_vid_name=output_name,
        fps=args.fps,
        skip_save_images=False,
    )

    return os.path.join(avatar.video_out_path, output_name + ".mp4")

@app.on_event("startup")
def load_avatars_into_memory():
    avatar_ids = ["avator_1"]
    for avatar_id in avatar_ids:
        video_path = f"data/avatars/{avatar_id}"
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=args.bbox_shift,
            batch_size=args.batch_size,
            preparation=False
        )
        avatar_cache[avatar_id] = avatar

@app.post("/generate-avatar/")
async def generate_avatar(
    avatar_id: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None)
):
    try:
        if not audio and not audio_url:
            raise HTTPException(status_code=400, detail="Either 'audio' file or 'audio_url' must be provided.")

        # Create temporary filenames
        temp_filename = f"{uuid.uuid4().hex}_temp.wav"
        audio_filename = f"{uuid.uuid4().hex}.wav"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)

        # Save uploaded or downloaded audio to temporary file
        if audio:
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)
        elif audio_url:
            r = requests.get(audio_url, stream=True)
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {audio_url}")
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Validate and convert audio
        if not validate_and_convert_audio(temp_path, audio_path):
            # Try FFmpeg as fallback
            if not convert_audio_with_ffmpeg(temp_path, audio_path):
                raise HTTPException(status_code=400, detail="Failed to process audio file. Please ensure it's a valid audio format.")

        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Run inference
        result_video_path = run_inference(avatar_id=avatar_id, audio_path=audio_path)

        # Read the video file and convert to hex
        try:
            with open(result_video_path, 'rb') as f:
                video_bytes = f.read()
                hex_data = video_bytes.hex()
                return JSONResponse(content={"video_hex": hex_data})
        finally:
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def start_ffmpeg_stream(fps: int, width: int, height: int):
    """
    Start FFmpeg process for streaming video output.
    Uses MPEGTS format which supports streaming to pipes.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",  # Input from stdin
        "-an",  # No audio
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-crf", "23",
        "-maxrate", "2M",
        "-bufsize", "4M",
        "-f", "mpegts",  # Use MPEGTS format for streaming
        "-"  # Output to stdout
    ]
    return subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE  # Capture stderr to prevent console spam
    )

@app.post("/stream-avatar/")
async def stream_avatar(
    avatar_id: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None)
):
    try:
        if not audio and not audio_url:
            raise HTTPException(status_code=400, detail="Provide either 'audio' file or 'audio_url'")

        # Create temporary filenames
        temp_filename = f"{uuid.uuid4().hex}_temp.wav"
        audio_filename = f"{uuid.uuid4().hex}.wav"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)

        # Save uploaded audio to temporary file
        if audio:
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)
        elif audio_url:
            r = requests.get(audio_url, stream=True)
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {audio_url}")
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Validate and convert audio
        if not validate_and_convert_audio(temp_path, audio_path):
            if not convert_audio_with_ffmpeg(temp_path, audio_path):
                raise HTTPException(status_code=400, detail="Failed to process audio file. Please ensure it's a valid audio format.")

        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Get or create avatar
        if avatar_id not in avatar_cache:
            avatar = Avatar(
                avatar_id=avatar_id,
                video_path=f"data/avatars/{avatar_id}",
                bbox_shift=args.bbox_shift,
                batch_size=args.batch_size,
                preparation=False
            )
            avatar_cache[avatar_id] = avatar
        else:
            avatar = avatar_cache[avatar_id]

        # Get frame dimensions from the avatar's first frame
        first_frame = avatar.frame_list_cycle[0]
        height, width = first_frame.shape[:2]
        
        ffmpeg_proc = start_ffmpeg_stream(args.fps, width, height)

        def generate():
            try:
                for frame in avatar.infer_stream(audio_path):
                    # Ensure frame is the right shape and type
                    if frame.dtype != 'uint8':
                        frame = frame.astype('uint8')
                    
                    # Write frame to ffmpeg stdin
                    try:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                        ffmpeg_proc.stdin.flush()
                    except BrokenPipeError:
                        print("FFmpeg process closed unexpectedly")
                        break
                    
                    # Read output from ffmpeg (non-blocking)
                    try:
                        chunk = ffmpeg_proc.stdout.read(8192)
                        if chunk:
                            yield chunk
                    except Exception as e:
                        print(f"Error reading from ffmpeg stdout: {e}")
                        break
                        
                # Close stdin to signal end of input
                try:
                    ffmpeg_proc.stdin.close()
                except:
                    pass
                
                # Read remaining output
                while True:
                    try:
                        chunk = ffmpeg_proc.stdout.read(8192)
                        if not chunk:
                            break
                        yield chunk
                    except Exception as e:
                        print(f"Error reading final chunks: {e}")
                        break
                    
            except Exception as e:
                print(f"Error in stream generation: {e}")
            finally:
                # Clean up
                try:
                    if ffmpeg_proc.stdin and not ffmpeg_proc.stdin.closed:
                        ffmpeg_proc.stdin.close()
                    ffmpeg_proc.terminate()
                    ffmpeg_proc.wait(timeout=5)
                except:
                    try:
                        ffmpeg_proc.kill()
                    except:
                        pass
                
                # Clean up audio file
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except:
                        pass

        # Use video/mp2t for MPEGTS streams
        return StreamingResponse(generate(), media_type="video/mp2t")

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
