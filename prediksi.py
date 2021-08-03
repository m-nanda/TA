import numpy as np

def myCountN(u):
  '''
  Fungsi ini untuk menghitung Panjang CPB/Rule.

  Input:
  > u: Rule yang ingin dicari panjangnya.

  Output:
  Panjang rule u.
  '''
  count = 1
  for i in u:
    if i == ',':
      count += 1
  return count


def sup(ket, support):
  '''
  Fungsi ini untuk melakukan normalisasi nilai
  support sedemikian hingga berada pada [0, 1]

  Input:
    > ket:
    > support:

  Output:
    Nilai support yang berada pada range [0, 1]
  '''
  ket = ket.lower()
  if ket== 'tidak berpotensi tinggi memiliki penyakit kardiovaskular':
    return support/27987
  else:
    return support/27989


def buat_df(hasil_hurim, sorting=['Utilitas','Support'], asc_dsc=[False, False]):
  '''
  Fungsi ini untuk mengonstruksi DataFrame dari hasil hurim
  yang akan digunakan sebagai database untuk melakukan prediksi.

  Input:
    > hasil_hurim: data hasil hurim yang telah diperoleh
    > sorting: pemilihan pengurutan rule yang diinginkan, bisa 
           berdasarkan Utilitas, Support, &/ Panjang CPB.
    > asc_dsc: untuk pemilihan pengurutan berdasarkan nilai 
           tertinggi / terenadah untuk setiap parameter 
           pada sorting.
  Output:
    DataFrame dengan kolom yang sesuai untuk proses prediksi, 
    terurut sesuai keperluan.
  '''
  import pandas as pd
  import numpy as np
  _util = []
  _sup = []
  _cpb = []
  _jml_cpb = 0
  _kelas = []

  for k in range(hasil_hurim.shape[0]):
    _cpb.append(hasil_hurim.iloc[k][0][0])
    _kelas.append(hasil_hurim.iloc[k][1])
    _util.append(hasil_hurim.iloc[k][0][1])
    _sup.append(hasil_hurim.iloc[k][0][2])

  hasil_rule = pd.DataFrame({'HURI': _cpb, 'Diagnosis': _kelas, 'Utilitas': _util, 'Support': _sup})

  hasil_rule['Panjang HURI'] = hasil_rule['HURI'].apply(lambda u: myCountN(u))
  hasil_rule['Support'] = hasil_rule.apply(lambda n: sup(n.Diagnosis, n.Support), axis=1)
  hasil_rule = hasil_rule[['Panjang HURI', 'HURI', 'Diagnosis', 'Utilitas', 'Support']]
  hasil_rule = hasil_rule.sort_values(sorting, ascending=asc_dsc).reset_index(drop=True)
  return hasil_rule


def prediksi_kardio(gejala, data_rule):
  '''
  Fungsi ini untuk melakukan prediksi penyakit kardiovaskular
  dari gejala yang muncul pada pasien berdasarkan rule/pattern
  yang didapat dari hasil hurim

  Input:
    > gejala:
    > data_rule:

  Output:
    ddsf
  '''
  i=0
  gejala = gejala.split(', ')    
  while True:
    r = data_rule.iloc[i]['HURI'].split(', ')        
    if all([True if item in gejala else False for item in r]):
      return data_rule.iloc[i]['HURI'], data_rule.iloc[i]['Utilitas'], data_rule.iloc[i]['Support'], data_rule.iloc[i]['Diagnosis']
    i+=1
    if i>=data_rule.shape[0]:
      return '-', '-', '-', 'tidak berpotensi tinggi memiliki penyakit kardiovaskular'

def pred_y(hasil):    
  return hasil[-1].lower()


def rule(hasil):
  return hasil[:-1]


ket = {'tidak berpotensi tinggi memiliki penyakit kardiovaskular':0, 'berpotensi tinggi memiliki penyakit kardiovaskular':1, '-':np.nan}


def hasil_metric(data):
  '''
  Fungsi ini untuk menghitung metric untuk hasil prediksi,
  yaitu: akurasi, presisi, recall, f1-score, dan confusion
  matriksnya.

  Input:
    > data: Data Tes yang sudah dilakukan prediksi.

  Output:
    Hasil akurasi, presisi, recall, f1-score, dan confusion
    matriks hasil prediksi.
  '''
  from sklearn.metrics import accuracy_score as akurasi
  from sklearn.metrics import precision_score as presisi
  from sklearn.metrics import recall_score as recall
  from sklearn.metrics import f1_score as f1
  from sklearn.metrics import confusion_matrix as cm

  #tn, fp, fn, tp = cm(data['y_aktual'], data['y_pred']).ravel()

  return akurasi(data['y_aktual'], data['y_pred']), presisi(data['y_aktual'], data['y_pred']), recall(data['y_aktual'], data['y_pred']), f1(data['y_aktual'], data['y_pred']), cm(data['y_aktual'], data['y_pred'])

def get_it_u(u, n):
  '''
  Fungsi ini untuk mengonversi umur menjadi kategori umur
  
  Input:
    > u: umur
    > n: n utk fuzzifikasi
  '''
  import pandas as pd
  import numpy as np
  import myFuzzy as my
  df_u = pd.DataFrame({'Umur': [u]})
  list_umur = [u]
	
  if n == 8:
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

    df_u['it_u'] = np.where(d_t_1 > d_t_2, 'Umur kurang dari 34 tahun', 
                  np.where(d_t_2 > d_t_3, 'Umur sekitar 31-38 tahun', 
                  np.where(d_t_3 > d_t_4, 'Umur sekitar 36-43 tahun',
                  np.where(d_t_4 > d_t_5, 'Umur sekitar 41-48 tahun',
                  np.where(d_t_5 > d_t_6, 'Umur sekitar 46-53 tahun',
                  np.where(d_t_6 > d_t_7, 'Umur sekitar 51-58 tahun',
                  np.where(d_t_7 > d_t_8, 'Umur sekitar 55-63 tahun',
                  'Umur lebih dari 60 tahun')))))))
    return df_u.iloc[0]['it_u']
		
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

    df_u['it_u'] = np.where(d_t_1 > d_t_2, 'Umur kurang dari 33 tahun', 
                  np.where(d_t_2 > d_t_3, 'Umur sekitar 31-37 tahun', 
                  np.where(d_t_3 > d_t_4, 'Umur sekitar 35-40 tahun',
                  np.where(d_t_4 > d_t_5, 'Umur sekitar 38-44 tahun',
                  np.where(d_t_5 > d_t_6, 'Umur sekitar 42-48 tahun',
                  np.where(d_t_6 > d_t_7, 'Umur sekitar 46-52 tahun',
                  np.where(d_t_7 > d_t_8, 'Umur sekitar 50-56 tahun',
                  np.where(d_t_8 > d_t_9, 'Umur sekitar 54-59 tahun',
                  np.where(d_t_9 > d_t_10, 'Umur sekitar 57-63 tahun',
                  'Umur lebih dari 61 tahun')))))))))

    return df_u.iloc[0]['it_u']


def get_it_bmi(m,t):
  '''
  Fungsi ini untuk mengonversi massa dan tinggi tubuh 
  menjadi kategori BMI.

  Input:
    > m: massa tubuh (kg)
    > t: tinggi tubuh (cm)

  Output:
    Hasil kategori BMI
  '''

  bmi = m / (t/100)**2

  if bmi<18.5:
    return 'BMI_Kurus'
  elif bmi <= 22.9:
    return 'BMI_Normal'
  elif bmi <= 24.9:
    return 'BMI_Overweight'
  elif bmi <= 29.9:
    return 'BMI_Obesitas'
  else:
    return 'BMI_Obesitas_II'

def get_it_bp(sis, dis):
  '''
  Fungsi ini untuk mengonversi tekanan sistol dan 
  tekanan diastol menjadi kategori tekanan darah

  Input:
    > sis: Tekanan Sistol
    > dis: Tekanan Diastol

  Output:
    Hasil kategori tekanan darah
  '''
  if sis<90 or dis<60: 
    return 'Tekanan darah rendah'
  elif 90<=sis<120 and 60<=dis<80:
    return 'Tekanan darah normal'
  elif 120<=sis<140 or 80<=dis<90:
    return 'Prahipertensi'
  elif 140<=sis<160 or 90<=dis<100:
    return 'Hipertensi tingkat 1'
  elif 160<=sis<180 or 100<=dis<120:
    return 'Hipertensi tingkat 2'
  elif sis>=180 or dis>=120:
    return 'Hipertensi krisis'
