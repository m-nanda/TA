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

  #---------------------------------#
  # Page layout
  ## Page expands to full width
#   st.set_page_config(page_title='Deteksi Penyakit Kardiovaskular Web App', 
#                     page_icon = "💗", 
#                     layout='wide')
  main_bg = "bg-6-dd.jpg"
  main_bg_ext = "jpg"

#   st.markdown(
#       f"""
#       <style>
#       .reportview-container {{
#           background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
#       }}
#       </style>
#       """,
#       unsafe_allow_html=True
#   )

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
    data_train_kardio = data_train[data_train['Diagnosis']=='berpotensi memiliki penyakit kardiovaskular'].copy().drop(columns=['Diagnosis'])
    data_train_kardio = data_train_kardio[['Gejala']]
    data_train_kardio['Gejala'] = data_train_kardio['Gejala'].apply(lambda u: upd_iu(u, iu))

    data_train_non_kardio = data_train[data_train['Diagnosis']=='Tidak berpotensi memiliki penyakit kardiovaskular'].copy().drop(columns=['Diagnosis'])
    data_train_non_kardio = data_train_non_kardio[['Gejala']]
    data_train_non_kardio['Gejala'] = data_train_non_kardio['Gejala'].apply(lambda u: upd_iu(u, iu))

    problem_kardio = hurim.UPTree(data_train_kardio, eu, min_util=min_utilitas, max_sup=int(max_support*len(data_train_kardio)))
    problem_non_kardio = hurim.UPTree(data_train_non_kardio, eu, min_util=min_utilitas, max_sup=int(max_support*len(data_train_non_kardio)))

    hasilnya_kardio = problem_kardio.solve_df()
    hasilnya_non_kardio = problem_non_kardio.solve_df()
    hasil_kardio = pd.DataFrame({'HURI': hasilnya_kardio, 'Diagnosis': ['berpotensi memiliki penyakit kardiovaskular']*len(hasilnya_kardio)})
    hasil_non_kardio = pd.DataFrame({'HURI': hasilnya_non_kardio, 'Diagnosis': ['tidak berpotensi memiliki penyakit kardiovaskular']*len(hasilnya_non_kardio)})
    hasil_hurim = pd.concat([hasil_kardio, hasil_non_kardio])
    hasil_hurim = buat_df(hasil_hurim, sorting=sort_)
    
    data_tes['Hasil'] = data_tes['Gejala'].apply(lambda n: prediksi_kardio(n, hasil_hurim))
    data_tes['Prediksi'] = data_tes['Hasil'].map(pred_y)
    data_tes['y_aktual'] = data_tes['Diagnosis'].apply(lambda u:u.lower()).replace(ket)
    data_tes['y_pred'] = data_tes['Prediksi'].replace(ket)
    
    acc, prec, rec, f1, cm = hasil_metric(data_tes)

    return hasil_hurim, acc, prec, rec, f1, cm

  #---------------------------------#
  # Deteksi Penyakit Kardiovaskular _Web App_ 🩺
  st.write("""  
  # Tab Admin
  Ini adalah bagian untuk Admin. Parameter untuk deteksi penyakit kardiovaskular menggunakan metode `HURIM` dapat disesuaikan dengan menu dibagian kiri.
  """)

  #---------------------------------#
  # Sidebar - Pemilihan Data
  with st.sidebar.header('1. Upload Data (CSV)'):
      uploaded_file = st.sidebar.file_uploader('Upload Data Rekam Medis (CSV)', type=["csv"])
      
  # Sidebar - Pemilihan N Untuk Fuzzy dan HURIM Params
  with st.sidebar.header('2. Parameter Preprocessing'):
      jumlah_n = st.sidebar.selectbox('Jumlah N untuk fuzzifikasi umur', [8, 10])

  with st.sidebar.header('3. Parameter HURIM'):
      min_utill = st.sidebar.slider('Batas Minimum Utilitas', 0.05, 0.5, 0.1, 0.01)
      max_supp = st.sidebar.slider('Batas Maksimum Support', 0.05, 0.5, 0.2, 0.01)
#       sort_HURI = st.sidebar.selectbox('Pengurutan HURI', ['Utilitas','Support'])
#       if sort_HURI is 'Utilitas':
#         sorting_HURI = ['Utilitas','Support']
#       else:
#         sorting_HURI = ['Support','Utilitas']
  sorting_HURI = ['Utilitas','Support']

  #---------------------------------#
  # Main panel

  # Displays the dataset
  st.subheader('1. Dataset')
  
  # Jika upload data
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')    
    df.drop(columns=['id'], inplace=True)
    st.markdown('Dataset Rekam Medis Awal')
    st.dataframe(df.head(10))        
    data, data_train, data_tes, u_ = dataPrep(df)
    hasil_hurim, acc, prec, rec, f1, cm = ekstraksiPola(data_train, min_utill, max_supp, data_tes, sorting_HURI)
    
    st.subheader('3. Ekstraksi Pola')
    st.dataframe(hasil_hurim)
    
    # Form deteksi
    with st.form('Form Deteksi Penyakit Kardiovaskular'):
      st.markdown("<h1 style='text-align: center; color:#f42563;'><b>Form Deteksi Penyakit Kardiovaskular</b></h1>", unsafe_allow_html=True)
      col1, col2, col3 = st.beta_columns(3)
      with col1:
        U = st.number_input('Masukkan umur anda:', step=0.1, min_value=0.0, format='%.f') #'{:.2f}'
        Jenis_Kelamin = st.selectbox('Pilih jenis kelamin anda', options=['Laki-Laki', 'Perempuan'])
        M = st.number_input('Masukkan massa tubuh anda:', step=0.1, min_value=0.0, format='%.f')
        T = st.number_input('Masukkan tinggi tubuh anda:', step=0.1, min_value=0.0, format='%.f') # format='%.f',
      with col2:
        TS = st.number_input('Masukkan tekanan sistol anda:', step=1, min_value=0) # format='%d',
        TD = st.number_input('Masukkan tekanan diastol anda:', step=1, min_value=0)          
        Kadar_Kolestrol = st.selectbox('Pilih kategori kadar kolesterol anda', options=['Kolestrol normal', 'Kolestrol jauh diatas normal', 'Kolestrol diatas normal'])
        Kadar_Glukosa = st.selectbox('Pilih kategori kadar glukosa anda', options=['Glukosa normal', 'Glukosa diatas normal', 'Glukosa jauh diatas normal'])
      with col3:
        Status_Perokok = st.selectbox('Pilih, apakah anda seorang perokok aktif atau bukan', options=['Bukan perokok aktif', 'Perokok aktif'])
        Status_Peminum_Alkohol = st.selectbox('Pilih, apakah anda Seorang peminum alkohol atau bukan', options=['Bukan peminum alkohol', 'Peminum alkohol'])
        Status_Kegiatan_Fisik = st.selectbox('Pilih, apakah anda aktif melakukan kegiatan fisik atau tidak', options=['Aktif berkegiatan fisik', 'Tidak aktif berkegiatan fisik']) 

      submitted = st.form_submit_button('Diagnosis')

      num_val = [U, M, T, TS, TD]
      if all([x>0 for x in num_val]):
        Umur = get_it_u(U, jumlah_n)
        Kategori_BMI = get_it_bmi(M,T)
        Kategori_Tekanan_Darah = get_it_bp(TS,TD)

        gejala = ', '.join([Umur, Jenis_Kelamin, Kategori_BMI, Kategori_Tekanan_Darah, #Umur, Kategori_BMI, Kategori_Tekanan_Darah, 
                            Kadar_Kolestrol, Kadar_Glukosa, Status_Perokok, 
                            Status_Peminum_Alkohol, Status_Kegiatan_Fisik])

        # Untuk mulai deteksi
        if submitted:
          st.write('Hasil deteksi:')
          data_Rq = hasil_hurim
          data_Rq_plus = data_Rq[data_Rq['Diagnosis'] == 'berpotensi memiliki penyakit kardiovaskular'].copy().drop(columns=['Panjang HURI']).sort_values(by=['Utilitas'], ascending=False)
          data_Rq_plus['Gejala'] = data_Rq_plus['HURI'].apply(lambda u: konvert(u, iu))
          data_Rq_plus = data_Rq_plus[['Gejala', 'Diagnosis', 'Utilitas', 'Support']]
          data_Rq_min = data_Rq[data_Rq['Diagnosis'] == 'tidak berpotensi memiliki penyakit kardiovaskular'].copy().drop(columns=['Panjang HURI']).sort_values(by=['Utilitas'], ascending=False)
          data_Rq_min['Gejala'] = data_Rq_min['HURI'].apply(lambda u: konvert(u, iu))
          data_Rq_min = data_Rq_min[['Gejala', 'Diagnosis', 'Utilitas', 'Support']]
          Diagnosis_, HURI_, Util_, Sup_, Confidence_, df_pred_ =  pred_sken_2(max_supp, min_utill, data_Rq_plus, data_Rq_min, gejala, eu)

#             msg = '{} (Presisi {}%)'.format(res[3], 72.86)
#             msg = '{} (Confidence: {})'.format(Diagnosis_, round(Confidence_,2))
          if Diagnosis_ == 'berpotensi memiliki penyakit kardiovaskular':
            msg = '{} (Confidence: {})'.format(Diagnosis_, round(Confidence_,2))
            st.error(msg)
          else:
            msg = '{} '.format(Diagnosis_)
            st.success(msg)

      # Jika ada field yang belum terisi
      else:
        var_num_ = ['Umur', 'Massa Tubuh', 'Tinggi Tubuh', 'Tekanan Sistol', 'Tekanan Diastol']
        index_0 = [i for i in range(len(num_val)) if num_val[i]<=0]
        field_0 = [var_num_[i] for i in index_0]
        field_0_ = ', '.join(field_0)
        if submitted:
          st.write('Silakan isi field berikut dengan benar terlebih dahulu:', field_0_)          
  # Jika menggunakan data sistem          
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
      
      # Form deteksi
      with st.form('Form Deteksi Penyakit Kardiovaskular'):        
        st.markdown("<h1 style='text-align: center; color:#f42563;'><b>Form Deteksi Penyakit Kardiovaskular</b></h1>", unsafe_allow_html=True)      
        col1, col2, col3 = st.beta_columns(3)
        with col1:
          U = st.number_input('Masukkan umur anda:', step=0.1, min_value=0.0, format='%.f') #'{:.2f}'
          Jenis_Kelamin = st.selectbox('Pilih jenis kelamin anda', options=['Laki-Laki', 'Perempuan'])
          M = st.number_input('Masukkan massa tubuh anda:', step=0.1, min_value=0.0, format='%.f')
          T = st.number_input('Masukkan tinggi tubuh anda:', step=0.1, min_value=0.0, format='%.f') # format='%.f',
        with col2:
          TS = st.number_input('Masukkan tekanan sistol anda:', step=1, min_value=0) # format='%d',
          TD = st.number_input('Masukkan tekanan diastol anda:', step=1, min_value=0)          
          Kadar_Kolestrol = st.selectbox('Pilih kategori kadar kolesterol anda', options=['Kolestrol normal', 'Kolestrol jauh diatas normal', 'Kolestrol diatas normal'])
          Kadar_Glukosa = st.selectbox('Pilih kategori kadar glukosa anda', options=['Glukosa normal', 'Glukosa diatas normal', 'Glukosa jauh diatas normal'])
        with col3:
          Status_Perokok = st.selectbox('Pilih, apakah anda seorang perokok aktif atau bukan', options=['Bukan perokok aktif', 'Perokok aktif'])
          Status_Peminum_Alkohol = st.selectbox('Pilih, apakah anda Seorang peminum alkohol atau bukan', options=['Bukan peminum alkohol', 'Peminum alkohol'])
          Status_Kegiatan_Fisik = st.selectbox('Pilih, apakah anda aktif melakukan kegiatan fisik atau tidak', options=['Aktif berkegiatan fisik', 'Tidak aktif berkegiatan fisik']) 

        submitted = st.form_submit_button('Diagnosis')
        
        num_val = [U, M, T, TS, TD]
        if all([x>0 for x in num_val]):
          Umur = get_it_u(U, jumlah_n)
          Kategori_BMI = get_it_bmi(M,T)
          Kategori_Tekanan_Darah = get_it_bp(TS,TD)

          gejala = ', '.join([Umur, Jenis_Kelamin, Kategori_BMI, Kategori_Tekanan_Darah, #Umur, Kategori_BMI, Kategori_Tekanan_Darah, 
                              Kadar_Kolestrol, Kadar_Glukosa, Status_Perokok, 
                              Status_Peminum_Alkohol, Status_Kegiatan_Fisik])

          # Untuk mulai deteksi
          if submitted:
            st.write('Hasil deteksi:')
            data_Rq = hasil_hurim
            data_Rq_plus = data_Rq[data_Rq['Diagnosis'] == 'berpotensi memiliki penyakit kardiovaskular'].copy().drop(columns=['Panjang HURI']).sort_values(by=['Utilitas'], ascending=False)
            data_Rq_plus['Gejala'] = data_Rq_plus['HURI'].apply(lambda u: konvert(u, iu))
            data_Rq_plus = data_Rq_plus[['Gejala', 'Diagnosis', 'Utilitas', 'Support']]
            data_Rq_min = data_Rq[data_Rq['Diagnosis'] == 'tidak berpotensi memiliki penyakit kardiovaskular'].copy().drop(columns=['Panjang HURI']).sort_values(by=['Utilitas'], ascending=False)
            data_Rq_min['Gejala'] = data_Rq_min['HURI'].apply(lambda u: konvert(u, iu))
            data_Rq_min = data_Rq_min[['Gejala', 'Diagnosis', 'Utilitas', 'Support']]
            Diagnosis_, HURI_, Util_, Sup_, Confidence_, df_pred_ =  pred_sken_2(max_supp, min_utill, data_Rq_plus, data_Rq_min, gejala, eu)

#             msg = '{} (Presisi {}%)'.format(res[3], 72.86)
#             msg = '{} (Confidence: {})'.format(Diagnosis_, round(Confidence_,2))
            if Diagnosis_ == 'berpotensi memiliki penyakit kardiovaskular':
              msg = '{} (Confidence: {})'.format(Diagnosis_, round(Confidence_,2))
              st.error(msg)
            else:
              msg = '{} '.format(Diagnosis_)
              st.success(msg)
              
        # Jika ada field yang belum terisi
        else:
          var_num_ = ['Umur', 'Massa Tubuh', 'Tinggi Tubuh', 'Tekanan Sistol', 'Tekanan Diastol']
          index_0 = [i for i in range(len(num_val)) if num_val[i]<=0]
          field_0 = [var_num_[i] for i in index_0]
          field_0_ = ', '.join(field_0)
          if submitted:
            st.write('Silakan isi field berikut dengan benar terlebih dahulu:', field_0_)
