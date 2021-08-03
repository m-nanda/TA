import streamlit as st
from multiApp import MultiApp
from apps import home, app_be, app_fe

st.set_page_config(page_title='Deteksi Penyakit Kardiovaskular Web App', 
                  page_icon = "💗", 
                  layout='wide')

app = MultiApp()

st.write("""
# Aplikasi Deteksi Penyakit Kardiovaskular 🩺
Aplikasi berbasis web ini digunakan sebagai implementasi deteksi penyakit kardiovaskular menggunakan metode `High Utility Rare Itemset Mining (HURIM)`. Aplikasi ini juga sebagai bagian dari Tugas Akhir.  
Untuk memulai, dapat silakan pilih tab menu yang tersedia:
* Home : Halaman awal.
* User : Halaman untuk pengguna umum, untuk melakukan deteksi saja.
* Admin : Halaman untuk pengguna khusus, dapat melakukan penyesuaian parameter dan melakukan deteksi.
""")


# Menambah tab menu
app.tambah_app('Home', home.app)
app.tambah_app('User', app_fe.app)
app.tambah_app('Admin', app_be.app)

# main app
app.run()
