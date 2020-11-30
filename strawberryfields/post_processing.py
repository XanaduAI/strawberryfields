import re
from copy import deepcopy
import numpy as np


def generate_wigner_chart(data, q):
    """Populates a RumoursUI chart dictionary with reduced
    Wigner function surface plot data.

    Args:
        chart (dict): chart dictionary requested by RumoursUI
        q (tuple[float]): minimum and maximum values of the q and p axes

    Returns:
        dict: a Plot.ly JSON-format surface plot
    """
    chart = {
            "data": [{"type": "scatter"}],
            "layout": {
                "scene":{
                    "xaxis": {},
                    "yaxis": {},
                    "zaxis": {}
                }
            }
        }
    chart['data'] = deepcopy(surfaceplotDefault['data'])

    p = q
    chart['data'][0]['x'] = q.tolist()
    chart['data'][0]['y'] = p.tolist()
    chart['data'][0]['z'] = data.tolist()

    chart['data'][0]['cmin'] = -1 / np.pi
    chart['data'][0]['cmax'] = 1 / np.pi
    # chart['data'][0]['zmin'] = chart['data'][0]['cmin']
    # chart['data'][0]['zmax'] = chart['data'][0]['cmax']

    for plot in chart['data']:
        plot['opacity'] = 0.95

    chart['layout']['paper_bgcolor'] = 'white'
    chart['layout']['plot_bgcolor'] = 'white'
    chart['layout']['scene']['bgcolor'] = 'white'
    # chart['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
    # chart['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
    # chart['layout']['scene']['bgcolor'] = 'rgba(0,0,0,0)'
    chart['layout']['font'] = {'color': textcolor}
    chart['layout']['font'] = {'color': textcolor}

    chart['layout']['scene']['xaxis']['title'] = 'x'
    chart['layout']['scene']['yaxis']['title'] = 'p'
    chart['layout']['scene']['zaxis']['title'] = 'W(x,p)'


    chart['layout']['scene']['xaxis'] = {
        'title' : 'x',
        'color' : textcolor
    }
    chart['layout']['scene']['yaxis'] = {
        'title' : 'p',
        # FIXME: Not working, not sure why
        # 'scaleanchor': 'x',
        # 'scaleratio' : 1,
        # 'constrain' : 'domain',
        'color' : textcolor,
        'gridcolor' : textcolor
    }

    # contour plot options
    chart['layout']['xaxis'] = {
        'title' : 'x',
        'color' : textcolor,
        'gridcolor' : textcolor
    }
    chart['layout']['yaxis'] = {
        'title' : 'p',
        'scaleanchor': 'x',
        'scaleratio' : 1,
        'constrain' : 'domain',
        'color' : textcolor,
        'gridcolor' : textcolor
    }

    return chart

textcolor = '#787878'

# Plot.ly default surface plot JSON
surfaceplotDefault = {
    'data': [{
        'cmin': -1 / np.pi,
        'cmax': 1 / np.pi,
        'contours': {'z': {'show': True}},
        'type': 'surface',
        'x': [],
        'y': [],
        'z': [],
        'colorscale': [
            [0.0, 'rgb(31,144,148)'],
            [0.5, 'rgb(255,255,255)'],
            [1.0, 'rgb(31,89,154)']
        ],
    }],
    'layout': {
        'width': 835,
        'height': 500,
        'margin': {
            'l': 5,
            'r': 5,
            'b': 5,
            't': 5,
            'pad': 4
        },
        'paper_bgcolor': 'transparent',
        'plot_bgcolor': 'transparent',
        'scene': {
            'bgcolor': 'transparent',
            'xaxis': {
                'gridcolor': textcolor,
                'gridwidth': 2,
            },
            'yaxis': {
                'gridcolor': textcolor,
                'gridwidth': 2,
            },
            'zaxis': {
                'autorange': True,
                'gridcolor': textcolor,
                'gridwidth': 2,
            }
        }
    },
    'config': {
        'modeBarButtonsToRemove': ['zoom2d','lasso2d','select2d','toggleSpikelines'],
        'displaylogo': False
    }
}
