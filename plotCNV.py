# dependencies
from dash import Dash, html, dash_table, callback, Output, Input, dcc
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# import data
# ----------------------------------------------------------------------------------------------------------------------
path_tirosh_dir = Path("/home/feilerwe/bench_data/dataloader/other/marker_genes__tirosh_group")
dict_tirosh_data = {p.stem: pd.read_csv(p) for p in path_tirosh_dir.glob("*.csv")}
dict_keys = {tag.replace("_", " "): tag for tag in dict_tirosh_data.keys()}
# ----------------------------------------------------------------------------------------------------------------------

app = Dash()
app.layout = [
    html.Div(className="background_div"),
    html.Div(id="cover_full_screen", className="main_div", style={},
             children=[html.Div(className="nav_bar", style={"width":"100vw"})]),
    html.Div(className="main_div smooth_transition", id="main_div",
        children=[
            html.Div(id="nav_bar", className="nav_bar smooth_transition", children=[
                html.Img(draggable="false", src="",
                         alt="InferCNV benchmark logo",
                         style={"width": "300px", "min-width": "250px", "margin-left": "0px", "filter": "invert(1)"}),
                dcc.Dropdown(options=["slim", "full screen"], value="slim" ,id="radio_pageWidth_settings", style={"width": "120px", "height":"35px"})
            ]),
            html.Div(id="behind_nav_bar_spacer"),
            dcc.Dropdown(options=list(dict_keys.keys()), value=list(dict_keys.keys())[0]),
            html.Div(style={"height":"50px"}),
        ])
]

# control page width settings
# -------------------------------------------------------------------------------------------------------------
@callback(
    Output(component_id="main_div", component_property="style"),
    Output(component_id="cover_full_screen", component_property="style"),
    Output(component_id="nav_bar", component_property="style"),
    Output(component_id="radio_pageWidth_settings", component_property="value"),
    Input(component_id="radio_pageWidth_settings", component_property="value")
)
def update_page(setting_page):
    width_main = "1200px"
    width_fig = "1100px"
    dict_cover_bg = {"opacity":"0", "transition":"none"}
    if setting_page == "full screen":
        dict_cover_bg = {"opacity": "1", "transition": "opacity 0s 0.7s"}
        width_main= "100vw"
        width_fig = "95vw"
    return {"width": width_main}, dict_cover_bg, {"width": width_main}, setting_page

# -------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False)
