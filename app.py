import gradio as gr
import requests
import websocket
import json
import uuid
import tempfile
import os

from cerebras.cloud.sdk import Cerebras

DEFAULT_MODAL_URL = "https://oktetod--example-comfyui-ui.modal.run/"
DEFAULT_NEGATIVE_PROMPT = "(static image:1.5), (still frame:1.4), (motionless:1.3), motion blur, blurry, low quality, bad quality, worst quality, jpeg artifacts, compression, watermark, text, signature, username, error, deformed, mutilated, extra limbs, bad anatomy, ugly, (fused fingers:1.2), (too many fingers:1.2), pixelated, low resolution"

CEREBRAS_API_KEY_PRIMARY = "csk-rh69dw68y9f2tffyvf82jdn3kkxv38tpvdwpwjjxx92hnfth"
CEREBRAS_API_KEY_BACKUP = "csk-j4dwwm65nhemrvhw3e86wennemkted2dpe9veme2nkymv48p"

workflow_template = {
  "3": { "inputs": { "seed": 418826476045691, "steps": 20, "cfg": 5, "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1, "model": [ "48", 0 ], "positive": [ "6", 0 ], "negative": [ "7", 0 ], "latent_image": [ "55", 0 ] }, "class_type": "KSampler" },
  "6": { "inputs": { "text": "DEFAULT_POSITIVE_PROMPT", "clip": [ "38", 0 ] }, "class_type": "CLIPTextEncode" },
  "7": { "inputs": { "text": "DEFAULT_NEGATIVE_PROMPT", "clip": [ "38", 0 ] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": [ "3", 0 ], "vae": [ "39", 0 ] }, "class_type": "VAEDecode" },
  "37": { "inputs": { "unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default" }, "class_type": "UNETLoader" },
  "38": { "inputs": { "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default" }, "class_type": "CLIPLoader" },
  "39": { "inputs": { "vae_name": "wan2.2_vae.safetensors" }, "class_type": "VAELoader" },
  "48": { "inputs": { "shift": 8.0, "model": [ "37", 0 ] }, "class_type": "ModelSamplingSD3" },
  "55": { "inputs": { "width": 1280, "height": 720, "length": 120, "batch_size": 1, "vae": [ "39", 0 ], "start_image": [ "56", 0 ] }, "class_type": "Wan22ImageToVideoLatent" },
  "56": { "inputs": { "image": "DEFAULT_IMAGE_NAME.jpeg" }, "class_type": "LoadImage" },
  "57": { "inputs": { "fps": 24, "images": [ "8", 0 ] }, "class_type": "CreateVideo" },
  "58": {
    "inputs": {
      "filename_prefix": "video/ComfyUI_Mobile",
      "format": "auto",
      "codec": "auto",
      "video-preview": "",
      "video": [ "57", 0 ]
    },
    "class_type": "SaveVideo"
  }
}

QUALITY_MAP = { "HD (720p)": (1280, 720), "SHD (1080p)": (1920, 1080) }
DURATION_MAP = { "5 Detik": 5, "8 Detik": 8, "10 Detik": 10 }
FPS_MAP = { "24 FPS": 24, "30 FPS": 30 }

cerebras_client_primary = None
cerebras_client_backup = None

if CEREBRAS_API_KEY_PRIMARY:
    try:
        cerebras_client_primary = Cerebras(api_key=CEREBRAS_API_KEY_PRIMARY)
    except Exception as e:
        print(f"Gagal inisialisasi Key Primer Cerebras: {e}")

if CEREBRAS_API_KEY_BACKUP:
    try:
        cerebras_client_backup = Cerebras(api_key=CEREBRAS_API_KEY_BACKUP)
    except Exception as e:
        print(f"Gagal inisialisasi Key Backup Cerebras: {e}")

cerebras_client_available = bool(cerebras_client_primary or cerebras_client_backup)

SYSTEM_PROMPT_ENHANCER = """Anda adalah seorang Sutradara film Horor dan ahli Urban Legend.
Tugas Anda adalah mengambil ide prompt dasar pengguna dan mengubahnya menjadi sebuah *shot list* sinematik yang mendetail untuk adegan 10 detik. Tujuannya adalah menciptakan atmosfer yang kental, mencekam, dan hidup dari gambar statis yang diberikan.

Ini adalah adegan 10 detik. JANGAN TAHAN DIRI. Kita perlu *beberapa lapis* gerakan untuk mengisi durasi tersebut.

FOKUS PADA 5 ELEMEN KUNCI:
1.  **Gerakan Subjek Utama:** Apa yang dilakukan subjek utama? Buatlah tidak nyaman. (Contoh: "Bukan hanya 'berdiri', tapi 'berdiri kaku, sedikit bergoyang maju mundur'", "kepala berputar sangat lambat", "senyum yang perlahan-lahan terbentuk").
2.  **Gerakan Atmosfer (Penting):** Ini membangun ketegangan. (Contoh: "kabut tebal yang merayap di lantai", "lampu jalan yang berkedip tidak menentu", "bayangan di dinding yang memanjang dan menari seolah hidup sendiri", "hujan deras memukul jendela").
3.  **Detail Horor (Maksimal):** Tambahkan BEBERAPA detail kecil untuk membuat adegan terasa 'salah' (uncanny). (Contoh: "mata berkedip ke samping (bukan ke bawah)", "retakan di dinding tampak bernapas atau melebar", "partikel debu tebal yang berputar melawan gravitasi", "nafas terlihat di udara dingin").
4.  **Sinematografi & Pencahayaan:** Bagaimana adegan ini direkam? (Contoh: "Gaya rekaman ditemukan (found footage) yang goyah", "pencahayaan *low-key* yang dramatis dengan bayangan pekat", "gaya Giallo (warna merah dan biru jenuh)", "pergerakan kamera yang lambat dan mengintai (creeping slow zoom in)").
5.  **Kualitas Sinematik:** Selalu akhiri dengan kualitas. (Contoh: "sangat detail, 4K, gerakan sinematik, fokus tajam, sinematografi horor").

ATURAN KETAT:
- Jangan melebihi 500 token untuk output prompt yang anda berikan
- JANGAN mengubah subjek inti atau ide utama pengguna (misal: "Wewe Gombel").
- JANGAN menambahkan teks percakapan (Contoh: "Tentu, ini...").
- HANYA KEMBALIKAN prompt yang telah disempurnakan, yang kaya dengan detail berlapis."""

def enhance_prompt(current_prompt):
    if not cerebras_client_available:
        raise gr.Error("Enhance Prompt gagal: Tidak ada CEREBRAS_API_KEY yang valid.")
    if not current_prompt:
        raise gr.Error("Silakan masukkan prompt terlebih dahulu sebelum di-enhance.")

    completion = None
    last_exception = None
    
    chat_payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_ENHANCER},
            {"role": "user", "content": current_prompt}
        ],
        "model": "gpt-oss-120b",
        "stream": False, 
        "max_completion_tokens": 500,
        "temperature": 7,
        "top_p": 8,
        "reasoning_effort": "high"
    }

    if cerebras_client_primary:
        try:
            print(f"Mencoba enhance prompt dengan key PRIMER...")
            completion = cerebras_client_primary.chat.completions.create(**chat_payload)
        except Exception as e:
            print(f"Key PRIMER GAGAL: {e}")
            last_exception = e

    if completion is None and cerebras_client_backup:
        try:
            print(f"Mencoba enhance prompt dengan key BACKUP...")
            completion = cerebras_client_backup.chat.completions.create(**chat_payload)
        except Exception as e:
            print(f"Key BACKUP GAGAL: {e}")
            last_exception = e
            
    if completion is None:
        raise gr.Error(f"Gagal menghubungi API Cerebras (kedua key gagal): {last_exception}")

    enhanced_text = completion.choices[0].message.content
    print(f"Enhanced prompt: {enhanced_text}")
    return enhanced_text.strip()

def get_comfy_output(
    modal_url, image_path, positive_prompt,
    quality_str, fps_str, duration_str
):
    
    print("--- Memulai Generate Baru ---")

    if not modal_url: raise gr.Error("Modal Server URL wajib diisi!")
    if image_path is None: raise gr.Error("Anda harus mengupload gambar input!")
    if not positive_prompt: raise gr.Error("Positive Prompt wajib diisi!")
    
    negative_prompt = DEFAULT_NEGATIVE_PROMPT

    if modal_url.endswith("/"): modal_url = modal_url[:-1]
    http_url = modal_url
    if not http_url.startswith("http"): http_url = "https://" + http_url
    ws_url = http_url.replace("https://", "wss://").replace("http://", "ws://")
    
    width, height = QUALITY_MAP.get(quality_str, (1280, 720))
    fps = FPS_MAP.get(fps_str, 24)
    duration_sec = DURATION_MAP.get(duration_str, 5)
    length = duration_sec * fps
    print(f"Pengaturan: {width}x{height} @ {fps}FPS, {duration_sec} detik ({length} frames)")

    try:
        print(f"Uploading image: {image_path}")
        files = {'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')}
        r_upload = requests.post(f"{http_url}/upload/image", files=files, data={"overwrite": "true"})
        r_upload.raise_for_status()
        image_name = r_upload.json()['name']
        print(f"Image uploaded as: {image_name}")

        client_id = str(uuid.uuid4())
        prompt_workflow = json.loads(json.dumps(workflow_template))
        prompt_workflow["6"]["inputs"]["text"] = positive_prompt
        prompt_workflow["7"]["inputs"]["text"] = negative_prompt
        prompt_workflow["56"]["inputs"]["image"] = image_name
        prompt_workflow["55"]["inputs"]["width"] = width
        prompt_workflow["55"]["inputs"]["height"] = height
        prompt_workflow["55"]["inputs"]["length"] = length
        prompt_workflow["57"]["inputs"]["fps"] = fps
        payload = {"prompt": prompt_workflow, "client_id": client_id}

        print("Queueing prompt...")
        requests.post(f"{http_url}/prompt", json=payload).raise_for_status()

        print(f"Connecting to WebSocket: {ws_url}/ws?clientId={client_id}")
        ws = websocket.create_connection(f"{ws_url}/ws?clientId={client_id}")
        
        video_info = None
        
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                
                if message['type'] == 'status':
                    print(f"Queue remaining: {message['data']['status']['exec_info']['queue_remaining']}")
                elif message['type'] == 'progress':
                    data = message['data']
                    print(f"Progress: {data['value']}/{data['max']}")

                elif message['type'] == 'executed':
                    if message['data']['node'] == "58":
                        print("Video generation complete, memproses output...")
                        output_data = message['data']['output']
                        
                        print(f"DEBUG: Menerima output mentah dari Node 58: {json.dumps(output_data, indent=2)}")

                        try:
                            video_info = output_data['images'][0]
                            print("INFO: Kunci 'videos' berhasil ditemukan.")
                        
                        except KeyError:
                            print("ERROR: Kunci 'videos' tidak ditemukan!")
                            try:
                                video_info = output_data['previews'][0]
                                print("INFO: Kunci 'videos' tidak ada, tapi 'previews' ditemukan. Menggunakan 'previews'.")
                            
                            except KeyError:
                                output_keys = ", ".join(output_data.keys())
                                raise gr.Error(
                                    "Struktur data output tidak dikenal. 'videos'/'previews' tidak ditemukan. "
                                    f"Kunci yang diterima: [{output_keys}]. "
                                    "Silakan periksa log runtime Kinsta Anda."
                                )
                        
                        ws.close()
                        break 
            else:
                pass 
        
        print("WebSocket ditutup.")

        filename = video_info['filename']
        subfolder = video_info.get('subfolder', '') 
        file_type = video_info.get('type', 'output') 
        
        video_url = f"{http_url}/view?filename={filename}&subfolder={subfolder}&type={file_type}"
        print(f"Downloading video from: {video_url}")
        
        r_video = requests.get(video_url)
        r_video.raise_for_status()

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
    except Exception as e:
        if isinstance(e, gr.Error):
            raise e
        print(f"An error occurred: {e}")
        raise gr.Error(f"Terjadi kesalahan: {e}")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📱 Antarmuka Mobile Wan 2.2 (Image-to-Video)")
    
    with gr.Column():
        gr.Markdown("### 1. Koneksi")
        modal_url_input = gr.Textbox(label="URL Server Modal Anda", value=DEFAULT_MODAL_URL)
        
        gr.Markdown("### 2. Input")
        image_input = gr.Image(label="Gambar Input", type="filepath", sources=["upload", "clipboard"])
        positive_prompt_input = gr.Textbox(label="Positive Prompt", placeholder="Contoh: a man playing guitar, retro cartoon style...", lines=3)
        
        enhance_btn = gr.Button("✨ Enhance Prompt (via Cerebras)", variant="secondary", interactive=cerebras_client_available)
        if not cerebras_client_available:
            gr.Markdown("<p style='color: red;'>Enhance dinonaktifkan. Tidak ada CEREBRAS_API_KEY yang ditemukan.</p>")
        
        gr.Markdown("### 3. Pengaturan Video")
        with gr.Row():
            quality_input = gr.Radio(label="Kualitas Video", choices=["HD (720p)", "SHD (1080p)"], value="HD (720p)")
            fps_input = gr.Radio(label="Frame Rate (FPS)", choices=["24 FPS", "30 FPS"], value="24 FPS")
        duration_input = gr.Radio(label="Durasi Video", choices=["5 Detik", "8 Detik", "10 Detik"], value="5 Detik")
        
        gr.Markdown("---")
        generate_btn = gr.Button("🚀 Generate Video", variant="primary")
        
        gr.Markdown("### 4. Hasil")
        video_output = gr.Video(label="Hasil Video", show_label=False)

    generate_btn.click(
        fn=get_comfy_output,
        inputs=[
            modal_url_input, image_input, positive_prompt_input,
            quality_input, fps_input, duration_input
        ],
        outputs=[video_output]
    )

    enhance_btn.click(
        fn=enhance_prompt,
        inputs=[positive_prompt_input],
        outputs=[positive_prompt_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
