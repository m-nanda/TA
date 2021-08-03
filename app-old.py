import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import preprocssing_data as prep
from preprocessing import fuzzifikasi_umur as fuzz
from preprocessing import bentuk_data_transaksi_medis as trans
from preprocessing import tambah_iu as upd_iu
from preprocessing import iu
from preprocessing import eu
import hurim
from prediksi import buat_df
from prediksi import prediksi_kardio
from prediksi import pred_y
from prediksi import ket
from prediksi import hasil_metric
import base64

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Deteksi Penyakit Kardiovaskular Web App', 
                   page_icon = "💗", 
                   layout='wide')
main_bg = "bg-6-dd.jpg"
main_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#---------------------------------#
# Preprocessing
def dataPrep(data_awal):
  data = data_awal.copy()  
  data = prep(data)
  data = fuzz(data, jumlah_n)
  umur_ = data['kategori umur'].unique()
  data, data_train, data_tes = trans(data, 0.2)
  
  st.subheader('2. _Preprocessing_ Data')
  st.markdown('Hasil Data _Train_ Setelah _Preprocessing_ Data')
  st.dataframe(data_train.head(5))
  
  st.markdown('Hasil Data Tes Setelah _Preprocessing_ Data')
  st.dataframe(data_tes.head(5))
  
  return data, data_train, data_tes, umur_

# Ekstraksi Pola
@st.cache(suppress_st_warning=True)
def ekstraksiPola(data_train, min_utilitas, max_support, data_tes, sort_):
  data_train_kardio = data_train[data_train['Diagnosa']=='Berpenyakit kardiovaskular'].copy().drop(columns=['Diagnosa'])
  data_train_kardio = data_train_kardio[['Gejala']]
  data_train_kardio['Gejala'] = data_train_kardio['Gejala'].apply(lambda u: upd_iu(u, iu))

  data_train_non_kardio = data_train[data_train['Diagnosa']=='Tidak Berpenyakit kardiovaskular'].copy().drop(columns=['Diagnosa'])
  data_train_non_kardio = data_train_non_kardio[['Gejala']]
  data_train_non_kardio['Gejala'] = data_train_non_kardio['Gejala'].apply(lambda u: upd_iu(u, iu))

  problem_kardio = hurim.UPTree(data_train_kardio, eu, min_util=min_utilitas, max_sup=int(max_support*len(data_train_kardio)))
  problem_non_kardio = hurim.UPTree(data_train_non_kardio, eu, min_util=min_utilitas, max_sup=int(max_support*len(data_train_non_kardio)))

  hasilnya_kardio = problem_kardio.solve_df()
  hasilnya_non_kardio = problem_non_kardio.solve_df()
  hasil_kardio = pd.DataFrame({'HURI': hasilnya_kardio, 'Diagnosa': ['berpenyakit kardiovaskular']*len(hasilnya_kardio)})
  hasil_non_kardio = pd.DataFrame({'HURI': hasilnya_non_kardio, 'Diagnosa': ['tidak berpenyakit kardiovaskular']*len(hasilnya_non_kardio)})
  hasil_hurim = pd.concat([hasil_kardio, hasil_non_kardio])
  hasil_hurim = buat_df(hasil_hurim, sorting=sort_)
  
  data_tes['Hasil'] = data_tes['Gejala'].apply(lambda n: prediksi_kardio(n, hasil_hurim))
  data_tes['Prediksi'] = data_tes['Hasil'].map(pred_y)
  data_tes['y_aktual'] = data_tes['Diagnosa'].apply(lambda u:u.lower()).replace(ket)
  data_tes['y_pred'] = data_tes['Prediksi'].replace(ket)
  
  acc, prec, rec, f1, cm = hasil_metric(data_tes)

  return hasil_hurim, acc, prec, rec, f1, cm

#---------------------------------#
st.write("""
# Deteksi Penyakit Kardiovaskular _Web App_ 🩺
Deteksi penyakit kardiovaskular menggunakan metode `High Utility Rare Itemset Mining (HURIM)`. _Web App_ ini sebagai bagian dari Tugas Akhir.
""")

#---------------------------------#
# Sidebar - Pemilihan Data
with st.sidebar.header('1. Upload Data (CSV)'):
    uploaded_file = st.sidebar.file_uploader('Upload Data Rekam Medis (CSV)', type=["csv"])
    
# Sidebar - Pemilihan N Untuk Fuzzy dan HURIM Params
with st.sidebar.header('2. Parameter Preprocessing'):
    jumlah_n = st.sidebar.selectbox('Jumlah N untuk fuzzifikasi umur', [8, 10])

with st.sidebar.header('3. Parameter HURIM'):
    min_utill = st.sidebar.slider('Batas Minimum Utilitas', 0.05, 0.5, 0.09, 0.01)
    max_supp = st.sidebar.slider('Batas Maksimum Support', 0.05, 0.5, 0.4, 0.01)
    sort_HURI = st.sidebar.selectbox('Pengurutan HURI', ['Utilitas','Support'])
    if sort_HURI is 'Utilitas':
      sorting_HURI = ['Utilitas','Support']
    else:
      sorting_HURI = ['Support','Utilitas']

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file, sep=';')
  df.drop(columns=['id'], inplace=True)
  st.markdown('Dataset Rekam Medis Awal')
  st.dataframe(df.head(10))        
  data, data_train, data_tes, u_ = dataPrep(df)
  hasil_hurim, acc, prec, rec, f1, cm = ekstraksiPola(data_train, min_utill, max_supp, data_tes, sorting_HURI)
  
  st.subheader('3. Ekstraksi Pola')
  st.dataframe(hasil_hurim)
  with st.form('Form Deteksi Penyakit Kardiovaskular'):
    st.markdown("<h1 style='text-align: center; color:#f42563;'><b>Form Deteksi Penyakit Kardiovaskular</b></h1>", unsafe_allow_html=True) ##0E6251

    col1, col2, col3 = st.beta_columns(3)

    with col1:
      Umur = st.selectbox('Pilih kategori umur anda pada', options=sorted(u_))#['Umur kurang dari 34 tahun', 'Umur sekitar 36-43 tahun', 'Umur sekitar 46-53 tahun', 'Umur sekitar 41-48 tahun', 'Umur sekitar 51-58 tahun', 'Umur sekitar 55-63 tahun', 'Umur lebih dari 60 tahun'])
      Kategori_BMI = st.selectbox('Pilih kategori BMI anda', options=['BMI_Kurus', 'BMI_Normal', 'BMI_Overweight', 'BMI_Obesitas', 'BMI_Obesitas_II'])      
      Kategori_Tekanan_Darah = st.selectbox('Pilih kategori tekanan darah anda', options=['Tekanan darah rendah', 'Tekanan darah normal', 'Prahipertensi', 'Hipertensi tingkat 1', 'Hipertensi tingkat 2',  'Hipertensi krisis'])
    with col2:
      Jenis_Kelamin = st.selectbox('Pilih jenis kelamin anda', options=['Laki-Laki', 'Perempuan'])
      Kadar_Kolestrol = st.selectbox('Pilih kategori kadar kolesterol anda', options=['Kolestrol normal', 'Kolestrol jauh diatas normal', 'Kolestrol diatas normal'])
      Kadar_Glukosa = st.selectbox('Pilih kategori kadar glukosa anda', options=['Glukosa normal', 'Glukosa diatas normal', 'Glukosa jauh diatas normal'])
    with col3:
      Status_Perokok = st.selectbox('Pilih, apakah anda seorang perokok aktif atau bukan', options=['Bukan perokok aktif', 'Perokok aktif'])
      Status_Peminum_Alkohol = st.selectbox('Pilih, apakah anda Seorang peminum alkohol atau bukan', options=['Bukan peminum alkohol', 'Peminum alkohol'])
      Status_Kegiatan_Fisik = st.selectbox('Pilih, apakah anda aktif melakukan kegiatan fisik atau tidak', options=['Aktif berkegiatan fisik', 'Tidak aktif berkegiatan fisik']) 

    submitted = st.form_submit_button('Diagnosa')
    
    if submitted:
      st.write('Gejala Anda:', gejala)
      st.write('Hasil deteksi:')
      res = prediksi_kardio(gejala, hasil_hurim)
      metrik = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
      nilai = [acc, prec, rec, f1]
      indeks = nilai.index(max(nilai))        
      msg = '{} ({} {}%)'.format(res[3], metrik[indeks], round(max(nilai)*100,2))
      if res[3] == 'berpenyakit kardiovaskular':
        st.error(msg)
      else:
        st.success(msg)

else:
  st.info('Silakan pilih data (Upload / Gunakan Data Sistem)')
  if st.checkbox('Klik untuk menggunakan data dari Sistem'):
#     df = pd.read_csv('https://github.com/m-nanda/TA/blob/main/Data/cardio_train_ind.csv?raw=true', sep=';')
    df = pd.read_csv('https://raw.githubusercontent.com/mns-037/tes/main/Data/cardio_train_ind.csv', sep=';')
    df.drop(columns=['id'], inplace=True)
    st.markdown('Dataset Rekam Medis Awal')
    st.dataframe(df.head(10))        
    data, data_train, data_tes, u_ = dataPrep(df)
    hasil_hurim, acc, prec, rec, f1, cm = ekstraksiPola(data_train, min_utill, max_supp, data_tes, sorting_HURI)

    st.subheader('3. Ekstraksi Pola')
    st.dataframe(hasil_hurim)
    
    with st.form('Form Deteksi Penyakit Kardiovaskular'):
      
      st.markdown("<h1 style='text-align: center; color:#f42563;'><b>Form Deteksi Penyakit Kardiovaskular</b></h1>", unsafe_allow_html=True)
    
      col1, col2, col3 = st.beta_columns(3)

      with col1:
        Umur = st.selectbox('Pilih kategori umur anda pada', options=sorted(u_))#['Umur kurang dari 34 tahun', 'Umur sekitar 36-43 tahun', 'Umur sekitar 46-53 tahun', 'Umur sekitar 41-48 tahun', 'Umur sekitar 51-58 tahun', 'Umur sekitar 55-63 tahun', 'Umur lebih dari 60 tahun'])
        Kategori_BMI = st.selectbox('Pilih kategori BMI anda', options=['BMI_Kurus', 'BMI_Normal', 'BMI_Overweight', 'BMI_Obesitas', 'BMI_Obesitas_II'])      
        Kategori_Tekanan_Darah = st.selectbox('Pilih kategori tekanan darah anda', options=['Tekanan darah rendah', 'Tekanan darah normal', 'Prahipertensi', 'Hipertensi tingkat 1', 'Hipertensi tingkat 2',  'Hipertensi krisis'])
      with col2:
        Jenis_Kelamin = st.selectbox('Pilih jenis kelamin anda', options=['Laki-Laki', 'Perempuan'])
        Kadar_Kolestrol = st.selectbox('Pilih kategori kadar kolesterol anda', options=['Kolestrol normal', 'Kolestrol jauh diatas normal', 'Kolestrol diatas normal'])
        Kadar_Glukosa = st.selectbox('Pilih kategori kadar glukosa anda', options=['Glukosa normal', 'Glukosa diatas normal', 'Glukosa jauh diatas normal'])
      with col3:
        Status_Perokok = st.selectbox('Pilih, apakah anda seorang perokok aktif atau bukan', options=['Bukan perokok aktif', 'Perokok aktif'])
        Status_Peminum_Alkohol = st.selectbox('Pilih, apakah anda Seorang peminum alkohol atau bukan', options=['Bukan peminum alkohol', 'Peminum alkohol'])
        Status_Kegiatan_Fisik = st.selectbox('Pilih, apakah anda aktif melakukan kegiatan fisik atau tidak', options=['Aktif berkegiatan fisik', 'Tidak aktif berkegiatan fisik']) 

      submitted = st.form_submit_button('Diagnosa')

      gejala = ', '.join([Umur, Kategori_BMI, Kategori_Tekanan_Darah, Jenis_Kelamin, 
                        Kadar_Kolestrol, Kadar_Glukosa, Status_Perokok, 
                        Status_Peminum_Alkohol, Status_Kegiatan_Fisik])

      if submitted:
        st.write('Gejala Anda:', gejala)
        st.write('Hasil deteksi:')
        res = prediksi_kardio(gejala, hasil_hurim)
        metrik = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
        nilai = [acc, prec, rec, f1]
        indeks = nilai.index(max(nilai))        
        msg = '{} ({} {}%)'.format(res[3], metrik[indeks], round(max(nilai)*100,2))
        if res[3] == 'berpenyakit kardiovaskular':
          st.error(msg)
        else:
          st.success(msg)
