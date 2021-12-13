import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import preprocssing_data as prep
from preprocessing import fuzzifikasi_umur as fuzz
from preprocessing import bentuk_data_transaksi_medis as trans
from preprocessing import tambah_iu as upd_iu
from preprocessing import iu, eu
import hurim
from prediksi import buat_df, prediksi_kardio, pred_y, ket, hasil_metric,get_it_u, get_it_bp, get_it_bmi, pred_sken_2, konvert, hitung_conf
import base64

def app():

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
  
  st.write("""  
  # Tab User
  Ini adalah bagian tab untuk User. Silakan mengisi kondisi anda yang sesuai, lalu klik tombol `Diagnosis` untuk mengetahui hasilnya.
  """)
  jumlah_n = 8
  
  with st.form('Form Deteksi Penyakit Kardiovaskular'):

    st.markdown("<h1 style='text-align: center; color:#f42563;'><b>Form Deteksi Penyakit Kardiovaskular</b></h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.beta_columns(3)

    with col1:
      U = st.number_input('Masukkan umur anda:', step=0.1, min_value=0.0, format='%.f') #'{:.2f}'
      Jenis_Kelamin = st.selectbox('Pilih jenis kelamin anda', options=['Laki-Laki', 'Perempuan'])
      M = st.number_input('Masukkan massa tubuh anda (kg):', step=0.1, min_value=0.0, format='%.f')
      T = st.number_input('Masukkan tinggi tubuh anda (cm):', step=0.1, min_value=0.0, format='%.f') # format='%.f',
    with col2:
      TS = st.number_input('Masukkan tekanan sistol anda (mmHg):', step=1, min_value=0) # format='%d',
      TD = st.number_input('Masukkan tekanan diastol anda (mmHg):', step=1, min_value=0)          
      Kadar_Kolestrol = st.selectbox('Pilih kategori kadar kolesterol anda', options=['Kolestrol normal', 'Kolestrol jauh diatas normal', 'Kolestrol diatas normal'])
      Kadar_Glukosa = st.selectbox('Pilih kategori kadar glukosa anda', options=['Glukosa normal', 'Glukosa diatas normal', 'Glukosa jauh diatas normal'])
    with col3:
      Status_Perokok = st.selectbox('Pilih, apakah anda seorang perokok aktif atau bukan', options=['Bukan perokok aktif', 'Perokok aktif'])
      Status_Peminum_Alkohol = st.selectbox('Pilih, apakah anda Seorang peminum alkohol atau bukan', options=['Bukan peminum alkohol', 'Peminum alkohol'])
      Status_Kegiatan_Fisik = st.selectbox('Pilih, apakah anda aktif melakukan kegiatan fisik atau tidak', options=['Aktif berkegiatan fisik', 'Tidak aktif berkegiatan fisik']) 

    submitted = st.form_submit_button('Diagnosis')
    
    val_num = [U, M, T, TS, TD]
    
    if all([x>0 for x in val_num]):
      Umur = get_it_u(U, jumlah_n)
      Kategori_BMI = get_it_bmi(M,T)
      Kategori_Tekanan_Darah = get_it_bp(TS,TD)      
      gejala = ', '.join([Umur, Kategori_BMI, Kategori_Tekanan_Darah, Jenis_Kelamin, 
                          Kadar_Kolestrol, Kadar_Glukosa, Status_Perokok, 
                          Status_Peminum_Alkohol, Status_Kegiatan_Fisik])
      if submitted:        
        st.write('Hasil deteksi:') 
        hasil_hurim = pd.read_csv('https://raw.githubusercontent.com/m-nanda/TA/main/default_huri_70.csv', sep=';')
#         st.dataframe(hasil_hurim.head(5))
#         st.write(hasil_hurim.columns)
        res = prediksi_kardio(gejala, hasil_hurim)
  
        data_Rq = pd.read_csv('https://raw.githubusercontent.com/m-nanda/TA/main/default_huri_70.csv', sep=';')
        data_Rq_plus = data_Rq[data_Rq['Diagnosis'] == 'berpotensi memiliki penyakit kardiovaskular'].copy().drop(columns=['Panjang HURI']).sort_values(by=['Utilitas'], ascending=False)
        data_Rq_plus['Gejala'] = data_Rq_plus['HURI'].apply(lambda u: konvert(u, iu))
        data_Rq_plus = data_Rq_plus[['Gejala', 'Diagnosis', 'Utilitas', 'Support']]
        data_Rq_min = data_Rq[data_Rq['Diagnosis'] == 'tidak berpotensi memiliki penyakit kardiovaskular'].copy().drop(columns=['Panjang HURI']).sort_values(by=['Utilitas'], ascending=False)
        data_Rq_min['Gejala'] = data_Rq_min['HURI'].apply(lambda u: konvert(u, iu))
        data_Rq_min = data_Rq_min[['Gejala', 'Diagnosis', 'Utilitas', 'Support']]
        Diagnosis_, HURI_, Util_, Sup_, Confidence_, df_pred_ =  pred_sken_2(0.2, 0.1, data_Rq_plus, data_Rq_min, gejala, eu)

#         msg = '{} (Presisi {}%)'.format(res[3], 72.86)
#         msg = '{} (Confidence: {})'.format(Diagnosis_, round(Confidence_,2))
        if Diagnosis_ == 'berpotensi memiliki penyakit kardiovaskular':
          msg = '{} (Confidence: {})'.format(Diagnosis_, round(Confidence_,2))
          st.error(msg)
        else:
          msg = '{}'.format(Diagnosis_)
          st.success(msg)
#         st.write('Gejala Anda:', gejala)
        
    else:
      var_num_ = ['Umur', 'Massa Tubuh', 'Tinggi Tubuh', 'Tekanan Sistol', 'Tekanan Diastol']
      index_0 = [i for i in range(len(val_num)) if val_num[i]<=0]
      field_0 = [var_num_[i] for i in index_0]
      field_0_ = ', '.join(field_0)
      if submitted:
        st.write('Silakan isi field berikut dengan benar terlebih dahulu:', field_0_)
