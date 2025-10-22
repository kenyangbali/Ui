import gradio as gr
import requests
import websocket
import json
import uuid
import tempfile
import os

# --- Konfigurasi ---

# 1. URL SERVER MODAL (Sesuai permintaan Anda)
DEFAULT_MODAL_URL = "https://forreplit0088--example-comfyui-ui.modal.run/"

# 2. PROMPT NEGATIF BARU (Lebih efektif untuk video "ngonten")
DEFAULT_NEGATIVE_PROMPT = "(static image:1.5), (still frame:1.4), (motionless:1.3), motion blur, blurry, low quality, bad quality, worst quality, jpeg artifacts, compression, watermark, text, signature, username, error, deformed, mutilated, extra limbs, bad anatomy, ugly, (fused fingers:1.2), (too many fingers:1.2), pixelated, low resolution"

# 3. WORKFLOW TEMPLATE (Berdasarkan comfyui_api_wan2_2_5B_i2v.json ANDA)
#    Peningkatan kualitas telah diterapkan:
#    - Node "55": width=1920, height=1080, length=300 (10 detik)
#    - Node "57": fps=30
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
      "text": "DEFAULT_POSITIVE_PROMPT", # Akan diganti oleh UI
      "clip": [ "38", 0 ]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": {
      "text": "DEFAULT_NEGATIVE_PROMPT", # Akan diganti oleh UI
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
      "width": 1920,      # DIUBAH: Kualitas 1080p
      "height": 1080,     # DIUBAH: Kualitas 1080p
      "length": 300,      # DIUBAH: 10 detik @ 30fps
      "batch_size": 1,
      "vae": [ "39", 0 ],
      "start_image": [ "56", 0 ]
    },
    "class_type": "Wan22ImageToVideoLatent"
  },
  "56": {
    "inputs": { "image": "DEFAULT_IMAGE_NAME.jpeg" }, # Akan diganti oleh UI
    "class_type": "LoadImage"
  },
  "57": {
    "inputs": {
      "fps": 30,          # DIUBAH: FPS Maksimal
      "images": [ "8", 0 ]
    },
    "class_type": "CreateVideo"
  },
  "58": {
    "inputs": {
      "filename_prefix": "video/ComfyUI_Mobile",
      "format": "auto",
      "codec": "auto",
      "video-preview": "",
      "video": [ "57", 0 ]
    },
    "class_type": "SaveVideo" # Kita akan memantau node ini
  }
}


# --- Fungsi Inti ---

def get_comfy_output(modal_url, image_path, positive_prompt, negative_prompt):
    """Fungsi utama untuk berinteraksi dengan API ComfyUI di Modal."""

    # 0. Menyiapkan URL
    if not modal_url:
        raise gr.Error("Modal Server URL wajib diisi!")
    if modal_url.endswith("/"):
        modal_url = modal_url[:-1]
    
    http_url = modal_url
    if not http_url.startswith("http"):
        http_url = "https://" + http_url
        
    ws_url = http_url.replace("https://", "wss://").replace("http://", "ws://")

    if image_path is None:
        raise gr.Error("Anda harus mengupload gambar input!")

    try:
        # 1. Upload Gambar
        print(f"Uploading image: {image_path}")
        files = {'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')}
        # Menambahkan parameter 'overwrite=true' agar tidak error jika ada file sama
        r_upload = requests.post(f"{http_url}/upload/image", files=files, data={"overwrite": "true"})
        r_upload.raise_for_status()
        image_data = r_upload.json()
        image_name = image_data['name']
        print(f"Image uploaded as: {image_name}")

        # 2. Menyiapkan Workflow JSON
        client_id = str(uuid.uuid4())
        
        # Deep copy template agar tidak menimpa
        prompt_workflow = json.loads(json.dumps(workflow_template)) 
        
        # Mengganti input
        prompt_workflow["6"]["inputs"]["text"] = positive_prompt
        prompt_workflow["7"]["inputs"]["text"] = negative_prompt
        prompt_workflow["56"]["inputs"]["image"] = image_name
        
        payload = {"prompt": prompt_workflow, "client_id": client_id}

        # 3. Mengirim (Queue) Prompt
        print("Queueing prompt...")
        r_prompt = requests.post(f"{http_url}/prompt", json=payload)
        r_prompt.raise_for_status()

        # 4. Mendengarkan WebSocket untuk Hasil
        print(f"Connecting to WebSocket: {ws_url}/ws?clientId={client_id}")
        ws = websocket.create_connection(f"{ws_url}/ws?clientId={client_id}")
        
        video_info = None
        
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

                # Menunggu node 'SaveVideo' (ID "58") selesai
                elif message['type'] == 'executed':
                    if message['data']['node'] == "58": # ID Node SaveVideo
                        print("Video generation complete, fetching video...")
                        output_data = message['data']['output']
                        
                        # Debug: print struktur output
                        print(f"Output data structure: {output_data}")
                        print(f"Output data keys: {output_data.keys()}")
                        
                        # SaveVideo node mengembalikan output dengan key berbeda
                        # Coba beberapa kemungkinan struktur
                        if 'gifs' in output_data:
                            video_info = output_data['gifs'][0]
                            print("Using 'gifs' key")
                        elif 'videos' in output_data:
                            video_info = output_data['videos'][0]
                            print("Using 'videos' key")
                        elif 'images' in output_data:
                            video_info = output_data['images'][0]
                            print("Using 'images' key")
                        else:
                            # Jika struktur tidak dikenali, ambil key pertama
                            if len(output_data.keys()) > 0:
                                first_key = list(output_data.keys())[0]
                                video_info = output_data[first_key][0]
                                print(f"Using fallback key: {first_key}")
                            else:
                                raise Exception("Output data kosong atau tidak memiliki key yang valid")
                        
                        ws.close()
                        break # Keluar dari loop
            else:
                pass # Abaikan data biner (preview)

        # Validasi video_info
        if video_info is None:
            raise Exception("Gagal mendapatkan informasi video dari output")

        # 5. Mengunduh Video
        filename = video_info['filename']
        subfolder = video_info.get('subfolder', '')  # gunakan .get() untuk menghindari KeyError
        file_type = video_info.get('type', 'output')  # default ke 'output'
        
        # Bangun URL dengan parameter yang benar
        if subfolder:
            video_url = f"{http_url}/view?filename={filename}&subfolder={subfolder}&type={file_type}"
        else:
            video_url = f"{http_url}/view?filename={filename}&type={file_type}"
        
        print(f"Downloading video from: {video_url}")
        
        r_video = requests.get(video_url)
        r_video.raise_for_status()

        # 6. Menyimpan video ke file temporer agar bisa ditampilkan Gradio
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
        print(f"KeyError: {e}")
        raise gr.Error(f"Struktur data output tidak sesuai. Key yang hilang: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
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
        # Input 1: URL Server Modal (dengan default value)
        modal_url_input = gr.Textbox(
            label="URL Server Modal Anda",
            value=DEFAULT_MODAL_URL
        )
        
        # Input 2: Gambar
        image_input = gr.Image(
            label="Gambar Input",
            type="filepath",
            sources=["upload", "clipboard"]
        )
        
        # Input 3: Prompt Positif
        positive_prompt_input = gr.Textbox(
            label="Positive Prompt",
            placeholder="Contoh: a man playing guitar, retro cartoon style...",
            lines=3
        )
        
        # Input 4: Prompt Negatif (dengan default baru)
        negative_prompt_input = gr.Textbox(
            label="Negative Prompt",
            value=DEFAULT_NEGATIVE_PROMPT,
            lines=5
        )
        
        # Tombol Submit
        generate_btn = gr.Button("🚀 Generate Video (1080p, 10 Detik)", variant="primary")
        
        # Output: Video
        video_output = gr.Video(label="Hasil Video")

    # Menghubungkan tombol ke fungsi
    generate_btn.click(
        fn=get_comfy_output,
        inputs=[
            modal_url_input,
            image_input,
            positive_prompt_input,
            negative_prompt_input
        ],
        outputs=[video_output]
    )

if __name__ == "__main__":
    # Menjalankan server Gradio
    demo.launch(server_name="0.0.0.0", server_port=8080)
