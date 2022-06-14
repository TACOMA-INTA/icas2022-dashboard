# streamlit_app.py

import streamlit as st
from PIL import Image
from tacoma.metrics import GetScores
import gdown
import numpy as np
from google.oauth2 import service_account
from gsheetsdb import connect
import pandas as pd
from pathlib import Path
from rep import plot_mesh,plot_hist
from scipy.io import loadmat

@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows


@st.cache
def download_drive_dataset(url,fname):
    """
    Download .npz from Google Drive

    :param url: url to the shared file
    :param fname: name of the file

    :returns path_to_file: Path to the file inside the system
    """
    download_path = Path("data_files")
    path_to_file = download_path/fname
    if not download_path.exists():
        download_path.mkdir()
    if not path_to_file.exists():
        gdown.download(url,output = str(path_to_file),quiet = True,fuzzy = True)
    return path_to_file

def npz_to_df(path:Path):
  
    try:
        data = loadmat(str(path)+".mat")
        x_name = "X_test"
    except:
        try:
            data = np.load(str(path)+".npz",allow_pickle=True)
            x_name = "X"
        except:
            raise ValueError(f"Data format{path} not supported")

    case = range(1,len(data[x_name])+1)
    df = pd.DataFrame()
    df["CaseID"] = case
    df["Mach"] = data[x_name][:,0]
    df["Alpha"] = data[x_name][:,1]
    return df,data
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
st.title("ICAS 2022") ## Titulo principal 
st.image(Image.open("assets/imgs/icas_logo.png"))
choices_query = run_query(f'SELECT * FROM "{sheet_url}"')
choices_query = pd.DataFrame(choices_query)
st.table(choices_query.drop(columns = ["url"]))
st.plotly_chart(plot_hist(choices_query.drop(columns = ["url"])))
choices = st.multiselect("Dataset to show",choices_query.dataset.to_list())
if not choices:
    st.stop()
dataset = choices_query.loc[choices_query.dataset.isin(choices)]
fname = download_drive_dataset(dataset.url.values[0],fname = f"{choices[0]}")
df,data = npz_to_df(fname)
if not fname.exists():
    st.stop()



##
# TODO: añadir tablas de estadísiticas generales
general_scores = GetScores(data["y_true"],data["y_pred"]).get_errors().mean()
r2_col,rmse_col,me_col = st.columns(3)
r2_col.metric("R2",round(general_scores.r2,2))
rmse_col.metric("Mean Squared Error",round(general_scores.mse,2))
me_col.metric("Maximum Error",round(general_scores.me,2))

## Elegir caso a visualizar


mesh = pd.read_pickle("data_files/mesh_garteur.pkl")
st.plotly_chart(plot_mesh(mesh,data["y_true"][0,:],data["y_pred"][0,:]))

#### GENERAL STATS

