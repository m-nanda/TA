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
	if ket== 'tidak berpenyakit kardiovaskular':
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

	hasil_rule = pd.DataFrame({'CPB': _cpb, 'Keterangan': _kelas, 'Utilitas': _util, 'Support': _sup})

	hasil_rule['Panjang CPB'] = hasil_rule['CPB'].apply(lambda u: myCountN(u))
	hasil_rule['Support'] = hasil_rule.apply(lambda n: sup(n.Keterangan, n.Support), axis=1)
	hasil_rule = hasil_rule[['Panjang CPB', 'CPB', 'Keterangan', 'Utilitas', 'Support']]
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
		r = data_rule.iloc[i]['CPB'].split(', ')        
		if all([True if item in gejala else False for item in r]):
			return data_rule.iloc[i]['CPB'], data_rule.iloc[i]['Utilitas'], data_rule.iloc[i]['Support'], data_rule.iloc[i]['Keterangan']
		i+=1
		if i>=data_rule.shape[0]:
			return '-', '-', '-', 'tidak berpenyakit kardiovaskular'
    
def pred_y(hasil):    
    return hasil[-1].lower()

	
def rule(hasil):
    return hasil[:-1]

	
ket = {'tidak berpenyakit kardiovaskular':0, 'berpenyakit kardiovaskular':1, '-':np.nan}


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