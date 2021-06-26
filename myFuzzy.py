'''
Library ini untuk membuat himpunan fuzzy dengan fungsi keanggotaan trapezoidal, triangular, &
gaussian
'''

import numpy as np

def TrapMf(interval, titik):
	'''
	Fungsi ini untuk membuat himpunan fuzzy dengan fungsi keanggotaan trapezoidal.
	
	Parameter:
	  Interval: himpunan nilai kontinu awal yang ingin dicari derajat keanggotaanya
	  titik: titik-titik A,B,C,D yang dibutuhkan untuk membuat fungsi keanggotaan trapezoidal	  
	
	Return:
	  Derajat keanggotaan dari setiap poin pada interval
	'''
	
	A = titik[0]
	B = titik[1]
	C = titik[2]
	D = titik[3]
	
	derajat_keanggotaan=[]
	
	def TMF(x):
		if (x<A):
			return 0
		elif (A<=x<B):
			return (x-A)/(B-A)
		elif (B<=x<=C):
			return 1
		elif (C<x<=D):
			return (D-x)/(D-C)
		else:
			return 0
			
	for x in interval:
		derajat_keanggotaan.append(TMF(x))
		
	return derajat_keanggotaan
	
	
def TriMf(interval, titik):
	'''
	Fungsi ini untuk membuat himpunan fuzzy dengan fungsi keanggotaan triangular.
	
	Parameter:
	  Interval: himpunan nilai kontinu awal yang ingin dicari derajat keanggotaanya
	  titik: titik-titik A,B,C yang dibutuhkan untuk membuat fungsi keanggotaan triangular
	
	Return:
	  Derajat keanggotaan dari setiap poin pada interval
	'''
	
	A = titik[0]
	B = titik[1]
	C = titik[2]    
    
	derajat_keanggotaan=[]    
    
	def TrMF(x):
		if (x<A or x>C):
			return 0
		elif (A<=x<=B):
			return (x-A)/(B-A)
		elif (B<x<=C):
			return (C-x)/(C-B)        		
        
	for x in interval:
		derajat_keanggotaan.append(TrMF(x))   

	return derajat_keanggotaan

def L_Mf(interval, titik): 
	'''
	Fungsi ini untuk membuat himpunan fuzzy terbuka ke kiri.
	
	Parameter:
	  Interval: himpunan nilai kontinu awal yang ingin dicari derajat keanggotaanya
	  titik: titik-titik A,B yang dibutuhkan untuk membuat fungsi terbuka ke kiri
	
	Return:
	  Derajat keanggotaan dari setiap poin pada interval
	'''
	
	A = titik[0]
	B = titik[1]        
    
	derajat_keanggotaan=[]    
    
	def LMF(x):
		if (x<=A):
			return 1
		elif (A<=x<=B):
			return (B-x)/(B-A)
		else:
			return 0        
        
	for x in interval:
		derajat_keanggotaan.append(LMF(x))   

	return derajat_keanggotaan

def R_Mf(interval, titik): 
	'''
	Fungsi ini untuk membuat himpunan fuzzy terbuka ke kanan.
	
	Parameter:
	  Interval: himpunan nilai kontinu awal yang ingin dicari derajat keanggotaanya
	  titik: titik-titik A,B yang dibutuhkan untuk membuat fungsi terbuka ke kanan
	
	Return:
	  Derajat keanggotaan dari setiap poin pada interval
	'''
	
	A = titik[0]
	B = titik[1]        
    
	derajat_keanggotaan=[]    
    
	def RMF(x):
		if (x<A):
			return 0
		elif (A<=x<=B):
			return (x-A)/(B-A)
		else:
			return 1
        
	for x in interval:
		derajat_keanggotaan.append(RMF(x))   

	return derajat_keanggotaan


def GaussMf(interval, titik):
	'''
	Fungsi ini untuk membuat himpunan fuzzy dengan fungsi keanggotaan Gaussian.
	
	Parameter:
	  Interval: himpunan nilai kontinu awal yang ingin dicari derajat keanggotaanya
	  titik: nilai sigma dan c yang dibutuhkan untuk membuat fungsi keanggotaan Gaussian. 
	
	Return:
	  Derajat keanggotaan dari setiap poin pada interval
	'''
	
	sigma = titik[0]
	c = titik[1]        
    
	derajat_keanggotaan=[]
    
	def GaussMF(x):
		return np.exp(-(x-c)**2/2*sigma**2) 
        
	for x in interval:
		derajat_keanggotaan.append(GaussMF(x))

	return derajat_keanggotaan