import os
import cv2
import numpy as np
from multiprocessing import Pool
import tqdm
import subprocess
from pathlib import Path
import shutil

def listFiles(path):
    return [os.path.join(dirName, filename)
            for dirName, _, fileList in os.walk(path)
            for filename in fileList if filename.lower().endswith(".png")]

def clean_results_directory(results_dir, final_video_name):
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        
        # Mantém apenas o vídeo final
        if item == final_video_name:
            continue
        
        # Remove diretórios
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        # Remove arquivos
        else:
            os.remove(item_path)

def ExtractFramesandAudioUsingFFMPEG(video_path, extracted_dir):
    os.makedirs(extracted_dir, exist_ok=True)
    TRIM_START = 6
    TRIM_END = 6
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    
    result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    total_duration = float(result.stdout.decode().strip())
    trimmed_duration = total_duration - TRIM_START - TRIM_END

    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-ss', str(TRIM_START), '-t', str(trimmed_duration), '-q:v', '2', f'{extracted_dir}/IMG%04d.png'], check=True)
    audio_path = os.path.join(extracted_dir, 'audio.aac')
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-ss', str(TRIM_START), '-t', str(trimmed_duration), '-vn', '-acodec', 'copy', audio_path], check=True)

def SaveFramesToVideoUsingFFMPEG(result_dir, processed_dir,extracted_dir):
    audio_path = os.path.join(extracted_dir, "audio.aac")
    subprocess.run(['ffmpeg', '-y', '-framerate', '60', '-i', f'{processed_dir}/IMG%04d.png', '-c:v', 'copy', f'{result_dir}/temp_video.mkv'], check=True)
    subprocess.run(['ffmpeg', '-y', '-i', f'{result_dir}/temp_video.mkv', '-i', audio_path, '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0', f'{result_dir}/Video_with_audio.mkv'], check=True)

def SaveVideoToH265(result_dir, video_name):
    subprocess.run(['ffmpeg', '-y', '-i', f'{result_dir}/Video_with_audio.mkv', '-vf', "scale='-2:min(1080,ih)'",'-r', '60', '-c:v', 'libx265', '-crf', '23', '-b:v', '40000k', f'{result_dir}/{video_name}'], check=True)

def SaveVideoToH264(result_dir, video_name):
    subprocess.run(['ffmpeg', '-y', '-i', f'{result_dir}/Video_with_audio.mkv', '-vf', "scale='-2:min(1080,ih)'",'-r', '60', '-c:v', 'libx264', '-crf', '23', '-b:v', '40000k', f'{result_dir}/{video_name}'], check=True)

def CropImageOnVerticalOrientation(Img):
    h, w = Img.shape[:2]
    target_ratio = 9 / 16
    if h / w > (16 / 9):
        new_w = w
        new_h = int(w * (16 / 9))
    else:
        new_h = h
        new_w = int(h * target_ratio)
    x_start = (w - new_w) // 2
    y_start = (h - new_h) // 2
    return Img[y_start:y_start + new_h, x_start:x_start + new_w]

def unsharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=2.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).round().astype(np.uint8)
    if threshold > 0:
        mask = np.abs(image - blurred) < threshold
        np.copyto(sharpened, image, where=mask)
    return sharpened

def apply_hdr_effect(img, detail_strength=1.5, smooth_sigma=75):
    """
    Aplica efeito HDR com controle de intensidade.
    - detail_strength: quanto mais alto, mais realce de detalhes (ex: 1.0 a 2.5)
    - smooth_sigma: suavidade do filtro bilateral (quanto maior, menos ruído realçado)
    """
    img_float = np.float32(img) / 255.0
    smooth = cv2.bilateralFilter(img_float, 9, smooth_sigma, smooth_sigma)
    detail = img_float - smooth
    hdr = img_float + detail_strength * detail
    hdr = np.clip(hdr, 0, 1)
    return np.uint8(hdr * 255)

def contrast(image, level):
    temp_img = np.int16(image)
    temp_img = temp_img * (level/127+1) - level
    temp_img = np.clip(temp_img, 0, 255)
    return np.uint8(temp_img)

def process_single_image(args):
    path, out_dir = args
    basename = os.path.basename(path)
    img = cv2.imread(path)
    if img is None:
        return
    result = CropImageOnVerticalOrientation(img)
    result = contrast(result, level=5) 
    #result = apply_hdr_effect(result, detail_strength=1.5, smooth_sigma=75)
    result = unsharp_mask(result, kernel_size=(3, 3), sigma=1.0, amount=4.0)
    cv2.imwrite(os.path.join(out_dir, basename), result)

def ProcessExtractedImages(extracted_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    image_paths = listFiles(extracted_dir)
    args = [(p, processed_dir) for p in image_paths if "IMG" in os.path.basename(p)]
    with Pool(processes=4) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_single_image, args), total=len(args)))

def process_video(video_path, base_result_dir):
    video_name = Path(video_path).stem + ".mp4"
    result_dir = os.path.join(base_result_dir, Path(video_path).stem)
    extracted_dir = os.path.join(result_dir, "Extracted")
    processed_dir = os.path.join(result_dir, "Processed")
    os.makedirs(extracted_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print(f" Processando: {video_name}")
    ExtractFramesandAudioUsingFFMPEG(video_path, extracted_dir)
    ProcessExtractedImages(extracted_dir, processed_dir)
    SaveFramesToVideoUsingFFMPEG(result_dir, processed_dir, extracted_dir)
    #SaveVideoToH265(result_dir, video_name)
    SaveVideoToH264(result_dir, video_name)
    print(f" Finalizado: {video_name}")
    clean_results_directory(result_dir, video_name)

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(project_dir, "Source_Videos")
    result_base = os.path.join(project_dir, "Results")

    os.makedirs(result_base, exist_ok=True)
    videos = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(".mp4")]

    for video in videos:
        process_video(video, result_base)

