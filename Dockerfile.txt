# 1. Gunakan base image Python yang ringan
FROM python:3.10-slim

# 2. Set direktori kerja di dalam container
WORKDIR /app

# 3. Install dependensi (requirement)
#    Kita install langsung di sini sesuai permintaan Anda.
RUN pip install --no-cache-dir \
    gradio==4.39.0 \
    requests==2.32.3 \
    websocket-client==1.8.0

# 4. Salin file aplikasi Anda ke dalam container
COPY app.py .

# 5. Expose port yang digunakan Gradio (default 7860)
EXPOSE 7860

# 6. Perintah untuk menjalankan aplikasi saat container dimulai
#    Ini akan mengeksekusi 'python app.py'
CMD ["python", "app.py"]
