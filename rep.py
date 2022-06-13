#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plotly.graph_objects as go
import numpy as np


def plot_mesh(mesh_data,y_true,y_pred):
    fig = go.Figure(data = [go.Mesh3d(x = mesh_data.X.values,
                                      y = mesh_data.Y.values,
                                      z = mesh_data.Z.values,
                                      colorscale = [[0,"blue"],
                                                    [1,"red"]
                                      ],
                                      colorbar_title = "Error",

                                      intensity = np.abs(y_true-y_pred),
                                      showlegend = False,
                                      intensitymode = "cell"
                                      )])
    return fig

