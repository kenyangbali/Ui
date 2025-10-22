# 1. Gunakan base image Python yang ringan
FROM python:3.10-slim

# 2. Set direktori kerja di dalam container
WORKDIR /app

# 3. Install dependensi (requirement)
#    Install versi terbaru yang tersedia
RUN pip install --no-cache-dir \
    gradio \
    requests \
    websocket-client

# 4. Salin file aplikasi Anda ke dalam container
COPY app.py .

# 5. Expose port yang digunakan Gradio (sesuai tuntutan Kinsta)
EXPOSE 8080

# 6. Perintah untuk menjalankan aplikasi saat container dimulai
#    Ini akan mengeksekusi 'python app.py'
CMD ["python", "app.py"]
