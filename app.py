import gradio as gr
import requests
import tempfile
import os
from cerebras.cloud.sdk import Cerebras
from typing import Literal

# URL default ke backend Modal T2V Anda
# Ganti ini dengan URL "modal serve" Anda setelah di-deploy
DEFAULT_MODAL_URL = "https://oktetod--wan22-t2v-ultra-fast-api-endpoint.modal.run"

# Prompt negatif default, bisa dikustomisasi
DEFAULT_NEGATIVE_PROMPT = "anime, cartoon, animated, illustration, drawing, painting, sketch, 3D render, CGI, plastic skin, doll-like, unrealistic skin texture, low quality, blurry, pixelated, jpeg artifacts, deformed hands, extra fingers, bad anatomy, static, still image, motionless"

# Kunci API Cerebras (jika ada)
CEREBRAS_API_KEY_PRIMARY = "csk-rh69dw68y9f2tffyvf82jdn3kkxv38tpvdwpwjjxx92hnfth"
CEREBRAS_API_KEY_BACKUP = "csk-j4dwwm65nhemrvhw3e86wennemkted2dpe9veme2nkymv48p"

# Inisialisasi Klien Cerebras
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

# Prompt sistem untuk Cerebras, dibuat lebih umum (bukan hanya horor)
SYSTEM_PROMPT_ENHANCER = """Anda adalah seorang Sutradara Sinematik.
Tugas Anda adalah mengambil ide prompt dasar pengguna dan mengubahnya menjadi *shot list* sinematik yang mendetail untuk adegan 10 detik.
Tujuannya adalah menciptakan adegan yang dinamis, kaya visual, dan hidup.

FOKUS PADA 4 ELEMEN KUNCI:
1.  **Gerakan Subjek Utama:** Apa yang dilakukan subjek utama? (Contoh: "wanita berjalan perlahan", "mata melihat ke kamera", "menari di bawah hujan").
2.  **Gerakan Atmosfer (Penting):** Apa yang terjadi di lingkungan? (Contoh: "kabut tebal yang merayap di lantai", "lampu neon berkedip", "bayangan memanjang", "hujan deras memukul jendela", "daun-daun berguguran").
3.  **Detail Visual:** Tambahkan detail kecil untuk membuat adegan terasa nyata. (Contoh: "partikel debu tebal yang berputar di bawah cahaya", "pantulan di genangan air").
4.  **Sinematografi & Pencahayaan:** Bagaimana adegan ini direkam? (Contoh: "Gaya rekaman sinematik", "pencahayaan dramatis", "pergerakan kamera yang lambat dan mengintai (creeping slow zoom in)", "sudut rendah (low-angle shot)").

ATURAN KETAT:
- JANGAN mengubah subjek inti atau ide utama pengguna.
- JANGAN menambahkan teks percakapan (Contoh: "Tentu, ini...").
- HANYA KEMBALIKAN prompt yang telah disempurnakan."""

def enhance_prompt(current_prompt):
    """Memanggil API Cerebras untuk menyempurnakan prompt pengguna."""
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
        "max_completion_tokens": 1500,
        "temperature": 0.7,
        "top_p": 0.8,
    }

    client_to_try = [cerebras_client_primary, cerebras_client_backup]
    
    for client in client_to_try:
        if client:
            try:
                print(f"Mencoba enhance prompt...")
                completion = client.chat.completions.create(**chat_payload)
                if completion:
                    break
            except Exception as e:
                print(f"Key GAGAL: {e}")
                last_exception = e
            
    if completion is None:
        raise gr.Error(f"Gagal menghubungi API Cerebras (semua key gagal): {last_exception}")

    enhanced_text = completion.choices[0].message.content
    
    if enhanced_text is None:
        print("ERROR: API Cerebras mengembalikan respons sukses namun 'content' adalah None.")
        raise gr.Error("Gagal 'enhance' prompt: Model tidak mengembalikan teks. Coba lagi.")

    print(f"Enhanced prompt: {enhanced_text}")
    return enhanced_text.strip()


def generate_video(
    modal_url, positive_prompt, negative_prompt,
    fps_str, seed, steps, cfg1, cfg2,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Memanggil backend Modal T2V (root/app.py) untuk menghasilkan video.
    """
    
    print("--- Memulai Generate Baru (T2V) ---")

    if not modal_url: 
        raise gr.Error("Modal Server URL wajib diisi!")
    if not positive_prompt: 
        raise gr.Error("Positive Prompt wajib diisi!")
    
    # Menghapus garis miring di akhir jika ada
    if modal_url.endswith("/"): 
        modal_url = modal_url[:-1]
    
    # Memastikan URL memiliki http/https
    if not modal_url.startswith("http"): 
        modal_url = "https://" + modal_url
        
    # Endpoint API backend T2V Anda
    api_url = f"{modal_url}/generate"

    # Mengurai FPS dari string "16 FPS" -> 16
    try:
        fps = int(fps_str.split(" ")[0])
    except:
        fps = 24 # Default

    # Membuat payload JSON yang sesuai dengan GenerateRequest Pydantic di backend
    payload = {
        "prompt": positive_prompt,
        "custom_negative_prompt": negative_prompt if negative_prompt else None,
        "fps": fps,
        "seed": int(seed),
        "num_inference_steps": int(steps),
        "guidance_scale": float(cfg1),
        "guidance_scale_2": float(cfg2),
    }

    print(f"Menghubungi: {api_url}")
    print(f"Payload: {payload}")
    progress(0, desc="Menghubungi server Modal...")

    try:
        # Backend (root/app.py) membutuhkan waktu 3-5 menit untuk merespons.
        # Kita perlu timeout yang sangat panjang.
        # Gradio mungkin timeout lebih dulu di sisi klien (default 60 detik).
        # Untuk produksi, Anda mungkin perlu streaming atau webhook.
        # Untuk saat ini, kita set timeout sisi server ke 10 menit (600 detik).
        with requests.post(api_url, json=payload, stream=True, timeout=600) as r:
            
            if r.status_code == 200:
                print("Sukses! Menerima file video, menyimpan ke disk...")
                progress(1, desc="✅ Sukses! Menyimpan video...")
                
                # Kita simpan konten (video .mp4) langsung ke file temporer
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    for chunk in r.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    
                    print(f"Video disimpan ke file temp: {temp_file.name}")
                    return temp_file.name
            else:
                # Jika backend mengembalikan error (misalnya 500)
                error_message = r.text
                try:
                    # Coba parse error JSON jika ada
                    error_json = r.json()
                    error_message = error_json.get("error", r.text)
                except:
                    pass
                
                print(f"HTTP Error: {r.status_code} - {error_message}")
                raise gr.Error(f"Gagal generate video (HTTP {r.status_code}): {error_message}")

    except requests.exceptions.ConnectionError as e:
        print(f"Koneksi Error: {e}")
        raise gr.Error(f"Gagal terhubung ke server Modal. Pastikan URL benar dan server berjalan: {e}")
    except requests.exceptions.ReadTimeout as e:
        print(f"Timeout Error: {e}")
        raise gr.Error("Gagal: Waktu tunggu habis. Backend T2V membutuhkan 3-5 menit. Server mungkin masih bekerja, tetapi Gradio kehabisan waktu tunggu.")
    except Exception as e:
        print(f"Terjadi error: {e}")
        raise gr.Error(f"Terjadi kesalahan: {e}")


# --- REFAKTOR ANTARMUKA GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Antarmuka Modal T2V Wan 2.2")
    gr.Markdown("UI ini memanggil backend `app.py` (Modal 8x H100) Anda. Resolusi (720x1280) dan Durasi (10 detik) sudah di-hardcode di backend.")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1. Koneksi")
            modal_url_input = gr.Textbox(
                label="URL Server Modal Anda", 
                value=DEFAULT_MODAL_URL,
                placeholder="https://anda--modal-app-name.modal.run"
            )
            
            gr.Markdown("### 2. Input Teks")
            positive_prompt_input = gr.Textbox(
                label="Positive Prompt", 
                placeholder="Contoh: a beautiful Indonesian woman walking in futuristic city, cinematic, 4k...", 
                lines=4
            )
            
            enhance_btn = gr.Button("✨ Enhance Prompt (via Cerebras)", variant="secondary", interactive=cerebras_client_available)
            if not cerebras_client_available:
                gr.Markdown("<p style='color: red;'>Enhance dinonaktifkan. Tidak ada CEREBRAS_API_KEY yang valid.</p>")

            negative_prompt_input = gr.Textbox(
                label="Negative Prompt (Opsional)", 
                placeholder="Gunakan prompt negatif universal dari backend jika dikosongi",
                value=DEFAULT_NEGATIVE_PROMPT,
                lines=3
            )
            
            gr.Markdown("### 3. Pengaturan Video")
            with gr.Row():
                # Opsi FPS harus sesuai dengan backend: Literal[16, 24]
                fps_input = gr.Radio(
                    label="Frame Rate (FPS)", 
                    choices=["16 FPS", "24 FPS"], 
                    value="24 FPS"
                )
                
                # Backend memiliki durasi tetap 10 detik, jadi kita nonaktifkan pilihan ini
                duration_input = gr.Textbox(
                    label="Durasi Video", 
                    value="10 Detik (Fixed by Backend)", 
                    interactive=False
                )
            
            with gr.Accordion("Pengaturan Lanjutan (Advanced)", open=False):
                with gr.Row():
                    # Tambahkan input untuk parameter backend lainnya
                    seed_input = gr.Number(label="Seed", value=42, precision=0)
                    steps_input = gr.Number(label="Steps", value=40, precision=0)
                with gr.Row():
                    cfg1_input = gr.Number(label="Guidance Scale 1 (CFG)", value=4.0)
                    cfg2_input = gr.Number(label="Guidance Scale 2 (CFG)", value=3.0)
            
            gr.Markdown("---")
            generate_btn = gr.Button("🚀 Generate Video (Estimasi 3-5 Menit)", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 4. Hasil")
            video_output = gr.Video(label="Hasil Video", show_label=False)

    # Hubungkan tombol 'Generate' ke fungsi baru
    generate_btn.click(
        fn=generate_video,
        inputs=[
            modal_url_input, positive_prompt_input, negative_prompt_input,
            fps_input, seed_input, steps_input, cfg1_input, cfg2_input
        ],
        outputs=[video_output]
    )

    # Tombol 'Enhance' tetap sama
    enhance_btn.click(
        fn=enhance_prompt,
        inputs=[positive_prompt_input],
        outputs=[positive_prompt_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
