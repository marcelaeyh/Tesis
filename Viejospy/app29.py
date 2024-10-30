import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import base64
from io import BytesIO
import cube_data29 as cd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

cube,Molines_A_df = cd.Cube_to_df('IRAS_15445-5449',29)
channel=[90,210]

data=np.sum(cube[channel[0]:channel[-1],:,:],axis=0)

# Crear la aplicación Dash con Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'IRAS 15445-5449 spw 29'
# Crear la figura usando px.imshow
fig = px.imshow(data, color_continuous_scale='Viridis')

fig.update_layout(
    coloraxis_colorbar=dict(title="Suma canales ["+str(channel[0])+','+str(channel[1])+']'),  # Mostrar la barra de color
    dragmode=False,                             # Desactivar el drag
    margin=dict(l=20, r=20, t=20, b=20)
)

indices = np.indices(data.shape)
customdata = np.stack([data, indices[0], indices[1]], axis=-1)


# Agregar contornos
contours = go.Contour(
    z=data,
    showscale=False,
    line_width=1,
    colorscale='gray',
    contours=dict(
        start=0.9,
        end=1.7,
        size=0.2,
        coloring='lines',
        showlabels = True, # show labels on contours
        labelfont = dict( # label font properties
                color = 'black'
            ),
        ),
    hoverinfo='skip',  # Desactivar el hover para los contornos
    name=''  # Eliminar el nombre del trazo de contornos
)


fig.add_trace(contours)

# Personalizar el hovertemplate
fig.update_traces(
    customdata=customdata,
    hovertemplate='<b>Valor Suma:</b> %{customdata[0]:.2f}<br>' +
                  '<b>x:</b> %{customdata[2]}<br>' +
                  '<b>y:</b> %{customdata[1]}<br>',
    name=''

)

app.layout = html.Div([
    html.H1('IRAS 15445-5449 spw 29', style={'textAlign': 'center', 'marginTop': '15px'}),
    
    dcc.Graph(id='image-graph', figure=fig, style={'height': '90vh', 'width': '90vw'}),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Imagen del Píxel"), close_button=True),
        dbc.ModalBody(html.Img(id='modal-image', src='', style={'width': '100%', 'height': '100%'})),
    ], id='modal', is_open=False, size='lg')  # Usa el tamaño grande de Bootstrap
])

def convert_matplotlib_to_base64(matplotlib_fig):
    buffer = BytesIO()
    matplotlib_fig.set_size_inches(10, 5)  # Cambiar el tamaño de la figura (en pulgadas)
    matplotlib_fig.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(matplotlib_fig)
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded_image}"



# Callback para mostrar la imagen en el modal
@app.callback(
    [Output('modal', 'is_open'),
     Output('modal-image', 'src')],
    [Input('image-graph', 'clickData')],
    [State('modal', 'is_open')]
)
def update_modal(clickData, is_open):
    if clickData:
        if not is_open:
            # Obtener las coordenadas del pixel clicado
            pixel_x = clickData['points'][0]['x']
            pixel_y = clickData['points'][0]['y']
            
            # Generar la figura de Matplotlib para el pixel clicado
            par,matplotlib_fig = cd.gauss_model(Molines_A_df,cube,'Pix_'+str(pixel_x)+'_'+str(pixel_y), channel,plot=True)
            
            # Convertir la figura de Matplotlib a base64
            img_src = convert_matplotlib_to_base64(matplotlib_fig)
            
            return True, img_src  # Abrir el modal y mostrar la imagen
        
    # Si no hay clickData o el modal está abierto, ciérralo
    return False, ''  # Cerrar el modal si no hay clicData o el modal está abierto

if __name__ == '__main__':
    app.run_server(debug=True, port=8081)



