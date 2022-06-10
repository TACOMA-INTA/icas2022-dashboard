# streamlit_app.py

import streamlit as st
import gdown
import numpy as np
from google.oauth2 import service_account
from gsheetsdb import connect
import pandas as pd
from pathlib import Path

@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows


@st.cache
def download_drive_dataset(url,fname):
    """
    Download .npz from Google Drive
    """
    download_path = Path("data_files")
    path_to_file = download_path/fname
    if not download_path.exists():
        download_path.mkdir()
    if not  path_to_file.exists():
        gdown.download(url,output = str(path_to_file),quiet = True,fuzzy = True)
    return path_to_file

def npz_to_df(data):
    case = range(1,len(data["X"])+1)
    df = pd.DataFrame()
    df["CaseID"] = case
    df["Mach"] = data["X"][:,0]
    df["Alpha"] = data["X"][:,1]
    return df
# Create a connection object.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
    ],
)
conn = connect(credentials=credentials)
# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 10 min.
sheet_url = st.secrets["private_gsheets_url"]
### DATA LOADING

choices_query = run_query(f'SELECT * FROM "{sheet_url}"')
choices_query = pd.DataFrame(choices_query)
choices = st.multiselect("Dataset to show",choices_query.dataset.to_list())
if not choices:
    st.stop()
dataset = choices_query.loc[choices_query.dataset.isin(choices)]
fname = Path(f"{choices[0]}.npz")
fname = download_drive_dataset(dataset.url.values[0],fname = f"{choices[0]}.npz")
data = np.load(fname,allow_pickle=True)
st.write(data.files)
df = npz_to_df(data)
st.write(df)


#### GENERAL STATS

