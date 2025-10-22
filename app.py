import gradio as gr
import requests
import websocket
import json
import uuid
import tempfile
import os

# --- Konfigurasi ---

# 1. URL SERVER MODAL ANDA
DEFAULT_MODAL_URL = "https://forreplit0088--example-comfyui-ui.modal.run/"

# 2. PROMPT NEGATIF
DEFAULT_NEGATIVE_PROMPT = "(static image:1.5), (still frame:1.4), (motionless:1.3), motion blur, blurry, low quality, bad quality, worst quality, jpeg artifacts, compression, watermark, text, signature, username, error, deformed, mutilated, extra limbs, bad anatomy, ugly, (fused fingers:1.2), (too many fingers:1.2), pixelated, low resolution"

# 3. WORKFLOW TEMPLATE (Berdasarkan comfyui_api_wan2_2_5B_i2v.json ANDA)
workflow_template = {
  "3": {
    "inputs": {
      "seed": 418826476045691,
      "steps": 20,
      "cfg": 5,
      "sampler_name": "uni_pc",
      "scheduler": "simple",
      "denoise": 1,
      "model": [ "48", 0 ],
      "positive": [ "6", 0 ],
      "negative": [ "7", 0 ],
      "latent_image": [ "55", 0 ]
    },
    "class_type": "KSampler"
  },
  "6": {
    "inputs": {
      "text": "DEFAULT_POSITIVE_PROMPT", # Akan diganti
      "clip": [ "38", 0 ]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": {
      "text": "DEFAULT_NEGATIVE_PROMPT", # Akan diganti
      "clip": [ "38", 0 ]
    },
    "class_type": "CLIPTextEncode"
  },
  "8": {
    "inputs": { "samples": [ "3", 0 ], "vae": [ "39", 0 ] },
    "class_type": "VAEDecode"
  },
  "37": {
    "inputs": {
      "unet_name": "wan2.2_ti2v_5B_fp16.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader"
  },
  "38": {
    "inputs": {
      "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
      "type": "wan",
      "device": "default"
    },
    "class_type": "CLIPLoader"
  },
  "39": {
    "inputs": { "vae_name": "wan2.2_vae.safetensors" },
    "class_type": "VAELoader"
  },
  "48": {
    "inputs": { "shift": 8.0, "model": [ "37", 0 ] },
    "class_type": "ModelSamplingSD3"
  },
  "55": {
    "inputs": {
      "width": 1280,      # Akan diganti
      "height": 720,      # Akan diganti
      "length": 120,      # Akan diganti
      "batch_size": 1,
      "vae": [ "39", 0 ],
      "start_image": [ "56", 0 ]
    },
    "class_type": "Wan22ImageToVideoLatent"
  },
  "56": {
    "inputs": { "image": "DEFAULT_IMAGE_NAME.jpeg" }, # Akan diganti
    "class_type": "LoadImage"
  },
  "57": {
    "inputs": {
      "fps": 24,          # Akan diganti
      "images": [ "8", 0 ]
    },
    "class_type": "CreateVideo"
  },
  "58": {
    "inputs": {
      "filename_prefix": "video/ComfyUI_Mobile",
      "format": "auto",
      "codec": "auto",
      # "video-preview": "",  <-- INI BARIS YANG DIHAPUS (PENYEBAB ERROR)
      "video": [ "57", 0 ]
    },
    "class_type": "SaveVideo" # Node yang dipantau
  }
}

# 4. PETA PENGATURAN (untuk menerjemahkan input UI)
QUALITY_MAP = {
    "HD (720p)": (1280, 720),
    "SHD (1080p)": (1920, 1080),
}
DURATION_MAP = {
    "5 Detik": 5,
    "8 Detik": 8,
    "10 Detik": 10,
}
FPS_MAP = {
    "24 FPS": 24,
    "30 FPS": 30,
}

# --- Fungsi Inti ---

def get_comfy_output(
    modal_url, 
    image_path, 
    positive_prompt, 
    negative_prompt, 
    quality_str, 
    fps_str, 
    duration_str
):
    """Fungsi utama untuk berinteraksi dengan API ComfyUI di Modal."""
    
    print("--- Memulai Generate Baru ---")

    # 0. Validasi Input
    if not modal_url:
        raise gr.Error("Modal Server URL wajib diisi!")
    if image_path is None:
        raise gr.Error("Anda harus mengupload gambar input!")
    if not positive_prompt:
        raise gr.Error("Positive Prompt wajib diisi!")

    # 1. Menyiapkan URL
    if modal_url.endswith("/"):
        modal_url = modal_url[:-1]
    
    http_url = modal_url
    if not http_url.startswith("http"):
        http_url = "https://" + http_url
        
    ws_url = http_url.replace("https://", "wss://").replace("http://", "ws://")
    
    # 2. Menerjemahkan Pengaturan UI
    width, height = QUALITY_MAP.get(quality_str, (1280, 720)) # Default 720p
    fps = FPS_MAP.get(fps_str, 24) # Default 24
    duration_sec = DURATION_MAP.get(duration_str, 5) # Default 5
    length = duration_sec * fps  # Hitung jumlah frame
    
    print(f"Pengaturan: {width}x{height} @ {fps}FPS, {duration_sec} detik ({length} frames)")

    try:
        # 3. Upload Gambar
        print(f"Uploading image: {image_path}")
        files = {'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')}
        r_upload = requests.post(f"{http_url}/upload/image", files=files, data={"overwrite": "true"})
        r_upload.raise_for_status()
        image_data = r_upload.json()
        image_name = image_data['name']
        print(f"Image uploaded as: {image_name}")

        # 4. Menyiapkan Workflow JSON
        client_id = str(uuid.uuid4())
        prompt_workflow = json.loads(json.dumps(workflow_template)) # Deep copy
        
        # --- MEMODIFIKASI WORKFLOW ---
        prompt_workflow["6"]["inputs"]["text"] = positive_prompt
        prompt_workflow["7"]["inputs"]["text"] = negative_prompt
        prompt_workflow["56"]["inputs"]["image"] = image_name
        
        # Terapkan pengaturan baru
        prompt_workflow["55"]["inputs"]["width"] = width
        prompt_workflow["55"]["inputs"]["height"] = height
        prompt_workflow["55"]["inputs"]["length"] = length
        prompt_workflow["57"]["inputs"]["fps"] = fps
        # --- SELESAI MODIFIKASI ---
        
        payload = {"prompt": prompt_workflow, "client_id": client_id}

        # 5. Mengirim (Queue) Prompt
        print("Queueing prompt...")
        r_prompt = requests.post(f"{http_url}/prompt", json=payload)
        r_prompt.raise_for_status()

        # 6. Mendengarkan WebSocket untuk Hasil
        print(f"Connecting to WebSocket: {ws_url}/ws?clientId={client_id}")
        ws = websocket.create_connection(f"{ws_url}/ws?clientId={client_id}")
        
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                
                if message['type'] == 'status':
                    status = message['data']['status']['exec_info']['queue_remaining']
                    print(f"Queue remaining: {status}")
                
                elif message['type'] == 'progress':
                    data = message['data']
                    print(f"Progress: {data['value']}/{data['max']}")

                elif message['type'] == 'executed':
                    # Menunggu node 'SaveVideo' (ID "58") selesai
                    if message['data']['node'] == "58": 
                        print("Video generation complete, fetching video...")
                        output_data = message['data']['output']
                        
                        # DEBUGGING (Boleh dihapus nanti)
                        print(f"DEBUG: Tipe data output_data: {type(output_data)}")
                        print(f"DEBUG: Isi output_data: {json.dumps(output_data, indent=2)}")

                        video_info = output_data['videos'][0] # Ini adalah baris yang crash (line 206)
                        ws.close()
                        break # Keluar dari loop
            else:
                pass # Abaikan data biner (preview)
        
        print("WebSocket ditutup.")

        # 7. Mengunduh Video
        filename = video_info['filename']
        subfolder = video_info['subfolder']
        
        video_url = f"{http_url}/view?filename={filename}&subfolder={subfolder}&type=output"
        print(f"Downloading video from: {video_url}")
        
        r_video = requests.get(video_url)
        r_video.raise_for_status()

        # 8. Menyimpan video ke file temporer agar bisa ditampilkan Gradio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(r_video.content)
            print(f"Video saved to temp file: {temp_file.name}")
            return temp_file.name

    except requests.exceptions.RequestException as e:
        print(f"HTTP Error: {e}")
        raise gr.Error(f"Gagal terhubung ke server Modal: {e}")
    except websocket.WebSocketException as e:
        print(f"WebSocket Error: {e}")
        raise gr.Error(f"Gagal terhubung ke WebSocket: {e}")
    except KeyError as e:
        print(f"KeyError: Kunci {e} tidak ditemukan di output JSON. Ini adalah akar masalahnya.")
        raise gr.Error(f"Terjadi kesalahan: Struktur data output tidak cocok. {e} tidak ditemukan.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise gr.Error(f"Terjadi kesalahan: {e}")


# --- Membangun Antarmuka (UI) Gradio ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📱 Antarmuka Mobile Wan 2.2 (Image-to-Video)
        UI ringan ini terhubung ke backend ComfyUI Anda yang dihosting di Modal.
        """
    )
    
    with gr.Column():
        
        gr.Markdown("### 1. Koneksi")
        modal_url_input = gr.Textbox(
            label="URL Server Modal Anda",
            value=DEFAULT_MODAL_URL
        )
        
        gr.Markdown("### 2. Input")
        image_input = gr.Image(
            label="Gambar Input",
            type="filepath",
            sources=["upload", "clipboard"]
        )
        positive_prompt_input = gr.Textbox(
            label="Positive Prompt",
            placeholder="Contoh: a man playing guitar, retro cartoon style...",
            lines=3
        )
        
        gr.Markdown("### 3. Pengaturan Video")
        with gr.Row():
            quality_input = gr.Radio(
                label="Kualitas Video",
                choices=["HD (720p)", "SHD (1080p)"],
                value="HD (720p)" # Default ke 720p (lebih cepat)
            )
            fps_input = gr.Radio(
                label="Frame Rate (FPS)",
                choices=["24 FPS", "30 FPS"],
                value="24 FPS" # Default ke 24
            )
        duration_input = gr.Radio(
            label="Durasi Video",
            choices=["5 Detik", "8 Detik", "10 Detik"],
            value="5 Detik" # Default ke 5 detik (tercepat)
        )
        
        gr.Markdown("### 4. (Opsional) Prompt Negatif")
        negative_prompt_input = gr.Textbox(
            label="Negative Prompt",
            value=DEFAULT_NEGATIVE_PROMPT,
            lines=5,
            show_label=False
        )
        
        gr.Markdown("---")
        
        # Tombol Submit
        generate_btn = gr.Button("🚀 Generate Video", variant="primary")
        
        # Output: Video
        gr.Markdown("### 5. Hasil")
        video_output = gr.Video(label="Hasil Video", show_label=False)

    # Menghubungkan tombol ke fungsi
    generate_btn.click(
        fn=get_comfy_output,
        inputs=[
            modal_url_input,
            image_input,
            positive_prompt_input,
            negative_prompt_input,
            quality_input,
            fps_input,
            duration_input
        ],
        outputs=[video_output]
    )

if __name__ == "__main__":
    # Menjalankan server Gradio di port 8080 (untuk Kinsta)
    demo.launch(server_name="0.0.0.0", server_port=8080)
