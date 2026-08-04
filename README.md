# Hand Mesh Hologram

Efek visual "hologram" mesh tangan real-time menggunakan webcam, **MediaPipe
Hands** untuk deteksi 2 tangan (21 landmark/tangan), dan render overlay
**OpenGL modern (GLSL shader)** — bukan `cv2.line`/`cv2.circle` biasa.

Fitur:

- Deteksi dua tangan sekaligus (kiri & kanan), 21 landmark per tangan.
- Mesh dinamis transparan (wireframe putih, alpha ~20%) dari **triangulasi
  Delaunay** atas gabungan titik kedua tangan → mesh otomatis "menjembatani"
  kedua tangan seperti jaring laba-laba holografik.
- Skeleton per-tangan (tulang jari) berwarna berbeda kiri/kanan.
- Titik landmark berupa dot glowing.
- Efek **glow** dibuat lewat fragment shader (radial/linear falloff per
  primitive), **additive blending**, dan **4x MSAA** untuk anti-aliasing.
- **Smoothing** dengan **One Euro Filter** (adaptif terhadap kecepatan
  gerakan) — bisa dimatikan (tekan `F`) untuk membandingkan dengan data
  mentah.
- HUD: FPS + label tangan + koordinat (x, y) ternormalisasi tiap titik.
- Target 30–60 FPS pada laptop modern (GPU terintegrasi sudah cukup, karena
  geometri sangat ringan: maksimum 42 titik).

## Struktur Proyek

```
hand_hologram/
├── main.py            # Entry point: env_check -> window/GL context -> capture loop
├── env_check.py        # Diagnosa Python/dependency otomatis + auto-fix mediapipe
├── setup.ps1            # Setup otomatis Windows (cari Python 3.11, buat venv, install)
├── setup.sh              # Setup otomatis Linux/macOS (idem)
├── hand_tracker.py     # Wrapper MediaPipe Hands (deteksi 2 tangan x 21 landmark)
├── mesh_generator.py   # Triangulasi Delaunay + skeleton connections
├── renderer.py         # Rendering moderngl/GLSL: background, mesh, skeleton, points, HUD
├── shader.py           # Semua source GLSL (#version 330 core) sebagai string Python
├── filters.py          # One Euro Filter & EMA untuk smoothing landmark
├── utils.py            # FPS counter, helper matematika, Config (semua parameter visual)
├── requirements.txt     # Versi dependency yang di-pin & sudah teruji kompatibel
└── README.md
```

## Instalasi

**Wajib Python 3.9 – 3.11.** MediaPipe (`mp.solutions.hands`, API yang
dipakai proyek ini) **tidak stabil di Python 3.12+** — kalau Anda pakai versi
lebih baru, akan muncul error `module 'mediapipe' has no attribute
'solutions'`. Cek versi Python Anda dulu:

```bash
python --version
```

### Opsi A — Setup Otomatis (disarankan)

Skrip ini otomatis mencari Python 3.9/3.10/3.11 yang terpasang, membuat
virtual environment, meng-upgrade pip, menginstall semua dependency dengan
versi yang benar, lalu menjalankan diagnosa (`env_check.py`) untuk
memastikan `mediapipe.solutions.hands` benar-benar berfungsi sebelum Anda
mencoba `python main.py`.

**Windows (PowerShell):**
```powershell
.\setup.ps1
```
Jika muncul error "execution policy", jalankan sekali (sebagai admin):
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Linux / macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

Setelah setup selesai:
```bash
# Windows
venv\Scripts\activate
python main.py

# Linux/macOS
source venv/bin/activate
python main.py
```

### Opsi B — Manual

```bash
py -3.11 -m venv venv           # Windows, pastikan pakai 3.11 bukan default
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
python env_check.py             # verifikasi semua dependency (termasuk mediapipe) OK
```

## Menjalankan

```bash
python main.py
```

`main.py` otomatis menjalankan pengecekan lingkungan (`env_check.py`) di
awal — kalau ada masalah (versi Python salah, dependency belum terinstall,
mediapipe rusak), Anda akan mendapat pesan yang jelas dan actionable,
**bukan traceback mentah**.

Webcam default (index 0) akan dibuka pada resolusi 1280x720. Jendela OpenGL
akan muncul menampilkan efek hologram secara real-time.

### Kontrol Keyboard

| Tombol      | Aksi                                                        |
|-------------|-------------------------------------------------------------|
| `ESC` / `Q` | Keluar                                                       |
| `H`         | Toggle HUD (FPS + koordinat)                                 |
| `C`         | Toggle label koordinat saja (FPS tetap tampil)               |
| `M`         | Toggle mirror (flip horizontal)                              |
| `F`         | Toggle One Euro Filter (bandingkan smoothed vs raw landmark) |

## Cara Kerja (Arsitektur Pipeline)

```
Webcam (BGR) → mirror + RGB
             → HandTracker.process()        [MediaPipe: hingga 2 tangan, 21 landmark]
             → LandmarkFilter.smooth_hand() [One Euro Filter per sumbu x,y,z]
             → MeshGenerator.build()        [Delaunay triangulation (scipy) + skeleton]
             → Renderer.render()            [moderngl: background, glow lines, glow points, HUD]
             → glfw.swap_buffers()
```

### Mengapa "OpenGL modern", bukan OpenCV drawing?

Semua elemen visual (garis wireframe, titik landmark, glow, HUD) digambar
sebagai **geometri GPU** (quad per garis/titik) yang dikirim ke **vertex
shader**, lalu efek glow & anti-aliasing dihitung per-pixel di **fragment
shader** (radial/linear falloff, `smoothstep`, additive blending). Ini
memberi hasil yang jauh lebih halus dan performan dibanding menggambar
ribuan pixel manual dengan `cv2.line`/`cv2.circle` setiap frame, dan lebih
mudah diberi efek neon/glow.

Kenapa quad manual, bukan `glLineWidth` biasa? Karena banyak driver *core
profile* modern membatasi `glLineWidth` ke 1.0 px, sehingga garis tebal +
glow dibangun sendiri di CPU sebagai dua segitiga (quad) per garis, dengan
atribut `side` (-1..1) yang dipakai fragment shader untuk falloff glow yang
mulus dan konsisten di semua platform.

### Delaunay mesh yang "menyambungkan kedua tangan"

`MeshGenerator` menggabungkan **seluruh titik dari kedua tangan** menjadi
satu awan titik, lalu menjalankan `scipy.spatial.Delaunay` sekali atas
gabungan tersebut. Karena triangulasi dihitung atas gabungan titik, tepi-tepi
segitiga secara alami akan menjembatani sisi-sisi tangan yang saling
berdekatan — inilah yang menghasilkan efek "jaring holografik" yang
menghubungkan kedua tangan seperti pada video referensi.

## Konfigurasi

Semua parameter visual (warna, alpha, lebar garis, kekuatan glow, brightness
background, dsb.) ada di satu tempat: `utils.py` → `class Config`. Ubah nilai
di sana untuk menyesuaikan tampilan tanpa menyentuh kode rendering.

Beberapa yang paling sering ingin diubah:

```python
mesh_alpha: float = 0.20            # transparansi mesh wireframe (~20%)
mesh_glow_power: float = 3.0        # makin besar -> glow makin tipis/fokus
skeleton_line_width_px: float = 3.2
point_radius_px: float = 6.5
one_euro_mincutoff: float = 1.2     # makin kecil -> makin halus tapi makin lag
one_euro_beta: float = 0.35         # makin besar -> makin responsif saat gerak cepat
```

## Troubleshooting

- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**:
  Python Anda kemungkinan 3.12+ (tidak didukung mediapipe 0.10.9), atau
  instalasi mediapipe korup. Jalankan diagnosa otomatis:
  ```bash
  python env_check.py
  ```
  Skrip ini akan mendeteksi versi Python, mengecek setiap dependency satu
  per satu, dan menawarkan reinstall otomatis untuk mediapipe. Kalau
  penyebabnya versi Python, jalankan `setup.ps1` (Windows) / `setup.sh`
  (Linux/macOS) untuk membuat virtual environment Python 3.11 otomatis.
  `main.py` sendiri sekarang juga menjalankan pengecekan ini di awal, jadi
  error akan tampil sebagai pesan jelas, bukan traceback.
- **`Could not open webcam`**: pastikan tidak ada aplikasi lain yang sedang
  memakai kamera, dan izin kamera untuk terminal/Python sudah diberikan
  (terutama macOS: System Settings → Privacy & Security → Camera).
- **Jendela gagal dibuat / error OpenGL 3.3**: perbarui driver GPU. Di Linux,
  pastikan driver GPU (Mesa/NVIDIA) mendukung OpenGL 3.3 core profile.
- **FPS rendah**: turunkan `capture_width`/`capture_height` di `Config`
  (mis. 960x540), atau set `glfw.swap_interval(0)` di `main.py` untuk
  menonaktifkan vsync saat benchmarking.
- **Instalasi `moderngl`/`glfw` gagal di Linux**: pastikan library sistem
  OpenGL/X11 terpasang (`libgl1-mesa-dev`, `libx11-dev`, dst. tergantung
  distro).
- **PowerShell menolak menjalankan `setup.ps1`**: jalankan sekali (sebagai
  admin) `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, lalu coba
  lagi.

## Pengembangan Lanjutan (Ide)

- Bloom multi-pass sungguhan (render-to-texture → bright-pass → gaussian
  blur → composite) untuk glow yang lebih lembut dari solusi single-pass
  saat ini.
- Gesture recognition (misal jarak ibu jari–telunjuk) untuk memicu efek
  partikel tambahan.
- Depth (z) landmark dipakai untuk mem-vary ketebalan/alpha garis agar mesh
  terasa lebih 3D.
