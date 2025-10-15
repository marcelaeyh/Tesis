import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import Ajuste 
import cube_to_df as ctdf

# Información de las líneas
info = {'13CO': [[90, 225], 0.1, [0.9, 1, 1.3, 1.5, 1.8], ' $^{13}CO$'],
        'SO2_4_3': [[90, 240], 0.1, [0.7, 0.8, 1, 1.2, 1.4], ' $SO_2$ $(4-3)$'],
        'SO2_21_21': [[50, 190], 0.1, [0.5, 0.6, 0.7, 0.9, 1, 1.1], ' $SO_2$ $(21-21)$']}

# Crear la aplicación Dash con Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'IRAS 15445-5449'

def create_figure(linea):
    path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_' + linea + '.fits'
    box = [205, 205, 305, 305]
    channel = info[linea][0]
    ruido = info[linea][1]
    cont = info[linea][2]
    latex = info[linea][3]
    

    cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)

    data = np.sum(cube[channel[0]:channel[-1], :, :], axis=0)
    data = np.flipud(data)

    fig = px.imshow(data, color_continuous_scale='Viridis')

    ts = 10
    xt = np.arange(100)[::ts]
    yt = np.arange(100)[::ts]

    x_tick_labels = np.round(coord[0][::ts], 5)
    y_tick_labels = np.round(coord[1][::ts], 5)

    fig.update_layout(
        xaxis=dict(
            title='J2000 RA offset [arcsec]',
            tickvals=xt,
            ticktext=x_tick_labels
        ),
        yaxis=dict(
            title='J2000 DEC offset [arcsec]',
            tickvals=yt,
            ticktext=-y_tick_labels
        )
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title="Suma canales [" + str(channel[0]) + ',' + str(channel[1]) + ']'),
        dragmode=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    indices = np.indices(data.shape)
    customdata = np.stack([data, np.flipud(indices[0]), indices[1]], axis=-1)

    contours = go.Contour(
        z=data,
        showscale=False,
        line_width=1,
        colorscale='gray',
        contours=dict(
            start=cont[0],
            end=cont[-1],
            size=0.2,
            coloring='lines',
            showlabels=True,
            labelfont=dict(
                color='black'
            ),
        ),
        hoverinfo='skip',
        name=''
    )

    fig.add_trace(contours)

    fig.update_traces(
        customdata=customdata,
        hovertemplate='<b>Valor Suma:</b> %{customdata[0]:.2f}<br>' +
                      '<b>x:</b> %{customdata[2]}<br>' +
                      '<b>y:</b> %{customdata[1]}<br>',
        name=''
    )

    return fig, Molines_A_df, cube, channel, ruido, latex

# Layout de la aplicación
app.layout = html.Div([
    html.H1(dcc.Markdown('IRAS 15445-5449', mathjax=True, style={'textAlign': 'center', 'marginTop': '15px'})),
    dcc.Dropdown(
        id='linea-dropdown',
        options=[{'label': linea, 'value': linea} for linea in info.keys()],
        value='13CO',
        clearable=False,
        style={'width': '50%', 'margin': '0 auto'}
    ),
    dcc.Graph(id='image-graph', style={'height': '90vh', 'width': '90vw'}),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Imagen del Píxel"), close_button=True),
        dbc.ModalBody(html.Img(id='modal-image', src='', style={'width': '100%', 'height': '100%'})),
    ], id='modal', is_open=False, size='xl')
])

def convert_matplotlib_to_base64(matplotlib_fig):
    buffer = BytesIO()
    matplotlib_fig.set_size_inches(15, 5)
    matplotlib_fig.savefig(buffer, format='png', bbox_inches='tight', dpi=200)
    plt.close(matplotlib_fig)
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded_image}"

# Callback para actualizar la figura según la línea seleccionada
@app.callback(
    Output('image-graph', 'figure'),
    Input('linea-dropdown', 'value')
)
def update_figure(linea):
    fig, Molines_A_df, cube, channel, ruido, latex = create_figure(linea)
    return fig

# Callback para mostrar la imagen en el modal
@app.callback(
    [Output('modal', 'is_open'),
     Output('modal-image', 'src')],
    [Input('image-graph', 'clickData')],
    [State('modal', 'is_open'),
     State('linea-dropdown', 'value')]
)
def update_modal(clickData, is_open, linea):
    if clickData:
        if not is_open:
            pixel_x = clickData['points'][0]['x']
            pixel_y = clickData['points'][0]['y']
            fig, Molines_A_df, cube, channel, ruido, _ = create_figure(linea)
            pixel_y = len(cube[channel[0]:channel[-1], :, :][0]) - 1 - pixel_y
            par, matplotlib_fig = Ajuste.gauss_model(Molines_A_df, cube, 'Pix_' + str(pixel_x) + '_' + str(pixel_y), channel, ruido, plot=True)
            img_src = convert_matplotlib_to_base64(matplotlib_fig)
            return True, img_src
    return False, ''

if __name__ == '__main__':
    app.run_server(debug=True, port=8081)
