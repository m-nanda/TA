def preprocssing_data(data):
  '''
  Fungsi untuk melakukan preprocessing data:
    > rename kolom ke bahasa indonesia
    > konversi umur ke tahun
    > reindex kolom
    > Menghapus outlier
    > Melakukan Feture Engineering

  Input:
    > Dataframe 

  Output:
    Dataframe yang siap diproses lebih lanjut
  '''

  import pandas as pd
  import numpy as np

  # Rename Col ke b.indo
  data = data.rename(columns={'u':'Umur (hari)', 'jk':'Jenis Kelamin', 't':'Tinggi',
                'm':'Massa Tubuh', 'ts':'Sistol', 'td':'Diastol',
                'k':'Kadar Kolestrol', 'g':'Kadar Glukosa', 'r':'Perokok Aktif',
                'a':'Peminum Alkohol', 'f':'Aktif Berkegiatan Fisik', 
                'y':'Diagnosis Penyakit Kardiovaskular'})

  # Convert Umur
  data['Umur (tahun-int)'] = data['Umur (hari)']/365
  data['Umur (tahun-int)'] = data['Umur (tahun-int)'].apply(lambda x: round(x, 0))
  data['Umur (tahun-int)'] = data['Umur (tahun-int)'].astype(int)
  data['Umur (tahun-float)'] = data['Umur (hari)']/365
  data['Umur (tahun-float)'] = data['Umur (tahun-float)'].apply(lambda x: round(x, 2))

  # Reindex Col
  data = data[['Umur (hari)', 'Umur (tahun-int)', 'Umur (tahun-float)',
         'Jenis Kelamin', 'Tinggi', 'Massa Tubuh', 'Sistol',
         'Diastol', 'Kadar Kolestrol', 'Kadar Glukosa', 'Perokok Aktif',
         'Peminum Alkohol', 'Aktif Berkegiatan Fisik', 'Diagnosis Penyakit Kardiovaskular']]

  # Remove Outlier
  data = data[(data['Sistol']>0) & (data['Diastol']>0)]

  #Feature Engineering
  data['BMI'] = data['Massa Tubuh'] / (data['Tinggi']/100)**2

  # Kategori tekanan darah
  def kategorikal_bp(sis, dis):
    # https://www.alodokter.com/memahami-klasifikasi-hipertensi-dan-faktor-risiko-yang-mempengaruhi
    # https://www.alodokter.com/hipotensi
    # https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings
    if sis<90 or dis<60: 
      return 1 #'Tekanan Darah Rendah'
    elif 90<=sis<120 and 60<=dis<80:
      return 2 #'Tekanan Darah Normal'
    elif 120<=sis<140 or 80<=dis<90:
      return 3 #'Prahipertensi'
    elif 140<=sis<160 or 90<=dis<100:
      return 4 #'Hipertensi tingkat 1'
    elif 160<=sis<180 or 100<=dis<120:
      return 5 #'Hipertensi tingkat 2'
    elif sis>=180 or dis>=120:
      return 6 #'Hipertensi krisis'

  # BMI
  def kategorikal_bmi(bmi):
    # http://p2ptm.kemkes.go.id/infographic-p2ptm/obesitas/klasifikasi-obesitas-setelah-pengukuran-imt
    if bmi<18.5:
      return 1 #'BMI_Kurus'
    elif bmi <= 22.9:
      return 2 #'BMI_Normal'
    elif bmi <= 24.9:
      return 3 #'BMI_Overweight'
    elif bmi <= 29.9:
      return 4 #'BMI_Obesitas'
    else:
      return 5 #'BMI_Obesitas_II'

  data['Kategori BMI'] = data['BMI'].apply(lambda t:kategorikal_bmi(t))
  data['Kategori Tekanan Darah'] = data.apply(lambda q: kategorikal_bp(q.Sistol, q.Diastol), axis=1)

  return data


def fuzzifikasi_umur(df, jumlah_N):
  '''
  Fungsi ini untuk melakukan fuzzifikasi terhadap atribut umur,
  menggunakan fungsi keanggotaan trapesium dengan pilihan N=8 
  atau N=10. Kemudian merubah semua atribut kategorikal yang 
  masih berbentuk angka menjadi kata/frase sesuai yang mudah
  dipahami.

  Input:
    > df: dataframe yang telah dilakukan proses preprocessing
    > jumlah_N: banyaknya pembagian (8 atau 10)

  Output:
    Dataframe dengan semua atributnya berupa kategorikal
    dengan nilainya berupa kata/frase.
  '''

  if jumlah_N not in [8, 10]:
    return 'Jumlah N yang dimasukkan tidak sesuai untuk fungsi ini. Pilih N=8 / N=10 saja.'

  import pandas as pd
  import numpy as np
  import myFuzzy as my
  df_hasil = df.copy()
  list_umur = df['Umur (tahun-float)'].to_list()
  kat = {        
      'Kategori BMI': {1:'BMI_Kurus', 2:'BMI_Normal', 3:'BMI_Overweight', 4:'BMI_Obesitas', 5:'BMI_Obesitas_II'},
      'Kategori Tekanan Darah': {1:'Tekanan darah rendah', 2:'Tekanan darah normal', 3:'Prahipertensi', 
                    4:'Hipertensi tingkat 1', 5:'Hipertensi tingkat 2', 6:'Hipertensi krisis'},
      'Jenis Kelamin': {1: 'Perempuan', 2: 'Laki-Laki'},
      'Kadar Kolestrol': {1: 'Kolestrol normal', 2: 'Kolestrol diatas normal', 3:'Kolestrol jauh diatas normal'},
      'Kadar Glukosa': {1: 'Glukosa normal', 2: 'Glukosa diatas normal', 3:'Glukosa jauh diatas normal'},
      'Perokok Aktif': {1: 'Perokok aktif', 0: 'Bukan perokok aktif'},
      'Peminum Alkohol': {1: 'Peminum alkohol', 0: 'Bukan peminum alkohol'},
      'Aktif Berkegiatan Fisik': {1: 'Aktif berkegiatan fisik', 0: 'Tidak aktif berkegiatan fisik'},
      'Diagnosis Penyakit Kardiovaskular': {1:'berpotensi memiliki penyakit kardiovaskular', 0:'Tidak berpotensi memiliki penyakit kardiovaskular'}
      }
  if jumlah_N == 8:
    r1 = [31.4, 33.8]
    r2 = [31.4, 33.8, 36.2, 38.6]
    r3 = [36.2, 38.6, 41.0, 43.4]
    r4 = [41.0, 43.4, 45.8, 48.2]
    r5 = [45.8, 48.2, 50.6, 53.0]
    r6 = [50.6, 53.0, 55.4, 57.8]
    r7 = [55.4, 57.8, 60.2, 62.6]
    r8 = [60.2, 62.6]

    d_t_1 = pd.DataFrame(my.L_Mf(list_umur, r1), columns=[''])
    d_t_2 = pd.DataFrame(my.TrapMf(list_umur, r2), columns=[''])
    d_t_3 = pd.DataFrame(my.TrapMf(list_umur, r3), columns=[''])
    d_t_4 = pd.DataFrame(my.TrapMf(list_umur, r4), columns=[''])
    d_t_5 = pd.DataFrame(my.TrapMf(list_umur, r5), columns=[''])
    d_t_6 = pd.DataFrame(my.TrapMf(list_umur, r6), columns=[''])
    d_t_7 = pd.DataFrame(my.TrapMf(list_umur, r7), columns=[''])
    d_t_8 = pd.DataFrame(my.R_Mf(list_umur, r8), columns=[''])

    df_hasil['kategori umur'] = np.where(d_t_1 > d_t_2, 'Umur kurang dari 34 tahun', 
                  np.where(d_t_2 > d_t_3, 'Umur sekitar 31-38 tahun', 
                  np.where(d_t_3 > d_t_4, 'Umur sekitar 36-43 tahun',
                  np.where(d_t_4 > d_t_5, 'Umur sekitar 41-48 tahun',
                  np.where(d_t_5 > d_t_6, 'Umur sekitar 46-53 tahun',
                  np.where(d_t_6 > d_t_7, 'Umur sekitar 51-58 tahun',
                  np.where(d_t_7 > d_t_8, 'Umur sekitar 55-63 tahun',
                  'Umur lebih dari 60 tahun')))))))
    df_hasil = df_hasil[['kategori umur', 'Kategori BMI', 'Kategori Tekanan Darah', 
               'Jenis Kelamin', 'Kadar Kolestrol', 'Kadar Glukosa', 
               'Perokok Aktif', 'Peminum Alkohol', 'Aktif Berkegiatan Fisik', 
               'Diagnosis Penyakit Kardiovaskular']]
    df_hasil = df_hasil.replace(kat)
    return df_hasil

  else:
    r1 = [30.89, 32.79]
    r2 = [30.89, 32.79, 34.68, 36.58]
    r3 = [34.68, 36.58, 38.47, 40.37]
    r4 = [38.47, 40.37, 42.26, 44.16]
    r5 = [42.26, 44.16, 46.05, 47.95]
    r6 = [46.05, 47.95, 49.84, 51.74]
    r7 = [49.84, 51.74, 53.63, 55.53]
    r8 = [53.63, 55.53, 57.42, 59.32]
    r9 = [57.42, 59.32, 61.21, 63.11]
    r10 = [61.21, 63.11]

    d_t_1 = pd.DataFrame(my.L_Mf(list_umur, r1), columns=[''])
    d_t_2 = pd.DataFrame(my.TrapMf(list_umur, r2), columns=[''])
    d_t_3 = pd.DataFrame(my.TrapMf(list_umur, r3), columns=[''])
    d_t_4 = pd.DataFrame(my.TrapMf(list_umur, r4), columns=[''])
    d_t_5 = pd.DataFrame(my.TrapMf(list_umur, r5), columns=[''])
    d_t_6 = pd.DataFrame(my.TrapMf(list_umur, r6), columns=[''])
    d_t_7 = pd.DataFrame(my.TrapMf(list_umur, r7), columns=[''])
    d_t_8 = pd.DataFrame(my.TrapMf(list_umur, r8), columns=[''])
    d_t_9 = pd.DataFrame(my.TrapMf(list_umur, r9), columns=[''])
    d_t_10 = pd.DataFrame(my.R_Mf(list_umur, r10), columns=[''])

    df_hasil['kategori umur'] = np.where(d_t_1 > d_t_2, 'Umur kurang dari 33 tahun', 
                  np.where(d_t_2 > d_t_3, 'Umur sekitar 31-37 tahun', 
                  np.where(d_t_3 > d_t_4, 'Umur sekitar 35-40 tahun',
                  np.where(d_t_4 > d_t_5, 'Umur sekitar 38-44 tahun',
                  np.where(d_t_5 > d_t_6, 'Umur sekitar 42-48 tahun',
                  np.where(d_t_6 > d_t_7, 'Umur sekitar 46-52 tahun',
                  np.where(d_t_7 > d_t_8, 'Umur sekitar 50-56 tahun',
                  np.where(d_t_8 > d_t_9, 'Umur sekitar 54-59 tahun',
                  np.where(d_t_9 > d_t_10, 'Umur sekitar 57-63 tahun',
                  'Umur lebih dari 61 tahun')))))))))

    df_hasil = df_hasil[['kategori umur', 'Kategori BMI', 'Kategori Tekanan Darah', 
               'Jenis Kelamin', 'Kadar Kolestrol', 'Kadar Glukosa', 
               'Perokok Aktif', 'Peminum Alkohol', 'Aktif Berkegiatan Fisik', 
               'Diagnosis Penyakit Kardiovaskular']]
    df_hasil = df_hasil.replace(kat)
    return df_hasil


def bentuk_data_transaksi_medis(df, persentase_data_tes):
  '''
  Fungsi ini untuk merubah dataset ke bentuk data transaksi medis

  Input:
    > df: Dataframe yang telah dilakukan preprocessing
    > persentase_data_tes: persentase jumlah data tes yang diinginkan
  Output:
    data transaksi medis full, data train transaksi medis full, dan
    data tes transaksi medis, 
  '''
  from sklearn.model_selection import train_test_split
  import pandas as pd
  import numpy as np

  df_trans_medis = df.copy()

  # Menggabungkan semua atribut yang termasuk gejala (X data / Antesenden)
  df_trans_medis['trans'] = df_trans_medis['kategori umur']+', '+df_trans_medis['Kategori BMI']+', '+df_trans_medis['Kategori Tekanan Darah']+', '+df_trans_medis['Jenis Kelamin']+', '+df_trans_medis['Kadar Kolestrol']+', '+df_trans_medis['Kadar Glukosa']+', '+df_trans_medis['Perokok Aktif']+', '+df_trans_medis['Peminum Alkohol']+', '+df_trans_medis['Aktif Berkegiatan Fisik']
  transaksi = df_trans_medis['trans'].to_list()

  # Membuat Y data / Konsekuen / Label
  kardio = df_trans_medis['Diagnosis Penyakit Kardiovaskular'].to_list() 

  # Membuat dataframe data transaksi medis
  d = {'Gejala': transaksi, 'Diagnosis': kardio}
  df_trans_medis = pd.DataFrame(data=d)

  # split data transaksi medis 
  data_trans_medis_train, data_trans_medis_tes = train_test_split(df_trans_medis, test_size = persentase_data_tes, random_state = 42)

  return df_trans_medis, data_trans_medis_train, data_trans_medis_tes

# iu=[
#     ['BMI_Kurus', 'Tekanan darah rendah', 'Perempuan', 'Kolestrol normal', 'Glukosa normal', 
#      'Bukan perokok aktif', 'Bukan peminum alkohol', 'Aktif berkegiatan fisik', 
#      'Umur kurang dari 34 tahun', 'Umur kurang dari 37 tahun', 'Umur kurang dari 33 tahun'], #1
#     ['BMI_Normal', 'Tekanan darah normal', 'Laki-Laki', 'Kolestrol diatas normal', 
#      'Glukosa diatas normal', 'Perokok aktif', 'Peminum alkohol', 'Tidak aktif berkegiatan fisik',
#      'Umur sekitar 31-38 tahun', 'Umur sekitar 33-41 tahun','Umur sekitar 31-37 tahun'], #2
#     ['BMI_Overweight', 'Prahipertensi', 'Kolestrol jauh diatas normal', 'Glukosa jauh diatas normal',
#      'Umur sekitar 36-43 tahun', 'Umur sekitar 37-45 tahun', 'Umur sekitar 35-40 tahun'], #3
#     ['BMI_Obesitas', 'Hipertensi tingkat 1', 
#      'Umur sekitar 41-48 tahun', 'Umur sekitar 41-49 tahun','Umur sekitar 38-44 tahun'], #4
#     ['BMI_Obesitas_II','Hipertensi tingkat 2', 
#      'Umur sekitar 46-53 tahun', 'Umur sekitar 45-53 tahun','Umur sekitar 42-48 tahun'], #5
#     ['Hipertensi krisis', 
#      'Umur sekitar 51-58 tahun','Umur sekitar 49-57 tahun','Umur sekitar 46-52 tahun'], #6
#     ['Umur sekitar 55-63 tahun','Umur sekitar 53-61 tahun','Umur sekitar 50-56 tahun'], #7
#     ['Umur lebih dari 60 tahun', 'Umur lebih dari 57 tahun','Umur sekitar 54-59 tahun'], #8
#     ['Umur sekitar 57-63 tahun'], #9
#     ['Umur lebih dari 61 tahun' ], #10
# ]

iu=[
    ['BMI_Kurus', 'Tekanan darah rendah', 'Perempuan', 'Kolestrol normal', 'Glukosa normal', 
     'Bukan perokok aktif', 'Bukan peminum alkohol', 'Aktif berkegiatan fisik', 
     'Umur kurang dari 34 tahun', 'Umur kurang dari 37 tahun', 'Umur kurang dari 33 tahun', 'U_1', 'U_1_', #1
     'BMI_Normal', 'Tekanan darah normal', 'Laki-Laki', 'Kolestrol diatas normal', 
     'Glukosa diatas normal', 'Perokok aktif', 'Peminum alkohol', 'Tidak aktif berkegiatan fisik',
     'Umur sekitar 31-38 tahun', 'Umur sekitar 33-41 tahun','Umur sekitar 31-37 tahun', 'U_2', 'U_2_' , #2
     'BMI_Overweight', 'Prahipertensi', 'Kolestrol jauh diatas normal', 'Glukosa jauh diatas normal',
     'Umur sekitar 36-43 tahun', 'Umur sekitar 37-45 tahun', 'Umur sekitar 35-40 tahun', 'U_3', 'U_3_', #3
     'BMI_Obesitas', 'Hipertensi tingkat 1', 
     'Umur sekitar 41-48 tahun', 'Umur sekitar 41-49 tahun','Umur sekitar 38-44 tahun', 'U_4', 'U_4_', #4
     'BMI_Obesitas_II','Hipertensi tingkat 2', 
     'Umur sekitar 46-53 tahun', 'Umur sekitar 45-53 tahun','Umur sekitar 42-48 tahun', 'U_5', 'U_5', #5
     'Hipertensi krisis', 
     'Umur sekitar 51-58 tahun','Umur sekitar 49-57 tahun','Umur sekitar 46-52 tahun', 'U_6', 'U_6_', #6
     'Umur sekitar 55-63 tahun','Umur sekitar 53-61 tahun','Umur sekitar 50-56 tahun', 'U_7', 'U_7_', #7
     'Umur lebih dari 60 tahun', 'Umur lebih dari 57 tahun','Umur sekitar 54-59 tahun', 'U_8', 'U_8_', #8
     'Umur sekitar 57-63 tahun', 'U_9', 'U_9_', #9
     'Umur lebih dari 61 tahun', 'U_10', 'U_10_', #10
    ]
]
eu = {
    'Aktif berkegiatan fisik': 0.32,          
    'Tidak aktif berkegiatan fisik': 1,       

    'Peminum alkohol': 1.56,                  
    'Bukan peminum alkohol': 1,               

    'Perokok aktif': 2.76,                    
    'Bukan perokok aktif': 1,                 

    'Glukosa diatas normal': 2.74,            
    'Glukosa jauh diatas normal':2.74,        
    'Glukosa normal': 1,                      

    'Kolestrol diatas normal': 1.37,          
    'Kolestrol jauh diatas normal': 1.12,     
    'Kolestrol normal': 1,                    

    'Laki-Laki': 1,                           
    'Perempuan': 0.46,                        

    'BMI_Kurus':2.42,                         
    'BMI_Normal':1,                           
    'BMI_Overweight':1,                       
    'BMI_Obesitas':0.64,                      
    'BMI_Obesitas_II':0.72,                   

    'Tekanan darah rendah':1,                 
    'Tekanan darah normal':1,                 
    'Prahipertensi':1.29,                     
    'Hipertensi tingkat 1':2.59,              
    'Hipertensi tingkat 2':2.59,              
    'Hipertensi krisis':2.59,                 

    'Umur kurang dari 34 tahun':1,
    'Umur sekitar 31-38 tahun':1,
    'Umur sekitar 36-43 tahun':1,
    'Umur sekitar 41-48 tahun':1,
    'Umur sekitar 46-53 tahun':1,
    'Umur sekitar 51-58 tahun':3.27,
    'Umur sekitar 55-63 tahun':3.27,
    'Umur lebih dari 60 tahun':3.14,

    'Umur kurang dari 33 tahun':1,
    'Umur sekitar 31-37 tahun':1,
    'Umur sekitar 35-40 tahun':1,
    'Umur sekitar 38-44 tahun':1,
    'Umur sekitar 42-48 tahun':1,
    'Umur sekitar 46-52 tahun':1,
    'Umur sekitar 50-56 tahun':3.27,
    'Umur sekitar 54-59 tahun':3.27,
    'Umur sekitar 57-63 tahun':3.14,
    'Umur lebih dari 61 tahun':3.14,
    }


def tambah_iu(list_item, iu):
  temp_item = list_item.split(', ')
  temp_list = []
  for item in temp_item:
    for i in range(len(iu)):
      if item in iu[i]:
        temp_list.append((item, i+1))        
        continue
  return temp_list
