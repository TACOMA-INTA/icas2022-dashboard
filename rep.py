#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st
import pandas as pd

def plot_mesh(mesh_data:pd.DataFrame,y_true,y_pred):
    """
    Plot mesh with the values of CP

    :params mesh_data: coordiantes of the points of the mesh
    :params y_true: true values of the CP 
    :params y_pred: predicted values of the CP
    """
    fig = go.Figure(data = [go.Mesh3d(x = mesh_data.X.values,
                                      y = mesh_data.Y.values,
                                      z = mesh_data.Z.values,
                                      colorscale = [[0,"white"],
                                                    [1,"red"]
                                      ],
                                      colorbar_title = "Error",

                                      intensity = np.abs(y_true-y_pred),
                                      showlegend = False,
                                      intensitymode = "cell"
                                      )])
    return fig

def plot_hist(data):
    fig = px.bar(data, x = "dataset", y="r2",range_y= [0.85,1])
    return fig