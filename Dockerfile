# Gunakan Python versi ringan
FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements dulu
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode dan model ke dalam container
COPY . .

# Beri tahu port yang digunakan Streamlit
EXPOSE 8501

# Perintah untuk menjalankan aplikasi saat container start
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]