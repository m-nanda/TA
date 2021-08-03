import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []

    def tambah_app(self, title, func):
        """
		Fungsi untuk menambah aplikasi.
        
		Parameter        
			> func: fungsi untuk merender aplikasi.
			> title: judul aplikasi yang akan ditampilkan di dropdown menu pilihan aplikasi.
        """
        self.apps.append({
            'title': title,
            'function': func
        })

    def run(self):
        # app = st.sidebar.radio(
        app = st.selectbox(
            'Pilih Tab Menu',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()
