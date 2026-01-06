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
                html.Img(draggable="false", src="/static/assets/weizmann_institute.png",
                         alt="Weizmann Institute Banner",
                         style={"width": "300px", "min-width": "250px", "margin-left": "0px", "filter": "invert(1)"}),
                dcc.Dropdown(options=["slim", "full screen"], value="slim" ,id="radio_pageWidth_settings", style={"width": "120px", "height":"35px"})
            ]),
            html.Div(id="behind_nav_bar_spacer"),
            html.Hr(),
            dcc.Dropdown(options=list(dict_keys.keys()), value=list(dict_keys.keys())[0], id="radio_tirosh_table"),
            html.Hr(),
            html.Div(style={"height":"50px"}),
            dash_table.DataTable(columns=[], data=[], id="input_tirosh_table", page_size=10, filter_action="native"),
            # records is essential so that the app runs
            html.Div(children=[
                dcc.Dropdown(options=["10 rows", "20 rows", "50 rows", "100 rows"], value="10 rows", id="radio_tirosh_len_table", style={"width": "120px", "height":"35px"}),
                html.Button(children="reset filter", id="clear_filter", n_clicks=0,
                            style={"width": "120px", "height":"36px", "margin-left":"10px", "border-color":"#ad1a38", "border-width":"thin", "border-radius":"5px", "background":"#e64e6d"})],
            style={"display":"flex", "margin-bottom":"30px", "justify-content":"right", "margin-top":"10px"}),
            html.Hr(),
            dcc.Dropdown(options=["balanced accuracy", "sensitivity", "specificity", "combined"], value="balanced accuracy",
                         id="radio_tirosh_figure"),
            dcc.Graph(figure={}, id="input_tirosh_figure", className="smooth_transition", style={"width": "1100px", "height": "700px"}),  # change one to change all (width)
            html.Hr()
        ])
]

# control page width settings
# -------------------------------------------------------------------------------------------------------------
@callback(
    Output(component_id="main_div", component_property="style"),
    Output(component_id="cover_full_screen", component_property="style"),
    Output(component_id="nav_bar", component_property="style"),
    Output(component_id="input_tirosh_figure", component_property="style"),
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
    return {"width": width_main}, dict_cover_bg, {"width": width_main}, {"width": width_fig}, setting_page

# -------------------------------------------------------------------------------------------------------------


# controls to build interactive table
# -------------------------------------------------------------------------------------------------------------
@callback(  # I forgot that this is the decorater for the function lol
    Output(component_id="input_tirosh_table", component_property="columns"),
    Output(component_id="input_tirosh_table", component_property="data"),
    Output(component_id="input_tirosh_table", component_property="page_size"),
    Input(component_id="radio_tirosh_table", component_property="value"),
    Input(component_id="radio_tirosh_len_table", component_property="value")
)
def update_table(chosen_table: str, len_table: str):
    key = dict_keys[chosen_table]
    df = dict_tirosh_data[key]
    list_columns = []
    dict_col_info = dict(df.dtypes)
    for col, type_col in dict_col_info.items():
        if str(type_col) == "object":
            type_col = "text"
        else:
            type_col = "numeric"
        list_columns.append({"name": col, "id": col, "type": type_col})
    return list_columns, df.to_dict("records"), int(len_table.split(" ")[0])

@callback(
    Output(component_id="input_tirosh_table", component_property="filter_query"),
    Output(component_id="clear_filter", component_property="n_clicks"),
    Input("clear_filter", component_property="n_clicks")
)
def reset_filter(n_clicks):
    if n_clicks != 0:
        return "", 0
# -------------------------------------------------------------------------------------------------------------


# controls to build interactive scatterplot
# -------------------------------------------------------------------------------------------------------------
@callback(
    Output(component_id="input_tirosh_figure", component_property="figure"),
    Input(component_id="radio_tirosh_table", component_property="value"),
    Input(component_id="radio_tirosh_figure", component_property="value")
)
def update_figure(chosen_table: str, chosen_size_param: str):
    key = dict_keys[chosen_table]
    df_selected = dict_tirosh_data[key].copy()
    df_selected["balanced accuracy"] = [round((sens + spec) / 2, 4) for sens, spec in
                                        zip(list(df_selected["sensitivity"]), list(df_selected["specificity"]))]
    first_col = list(df_selected.columns)[0]
    if chosen_size_param == "combined":
        chosen_size_param = "combined_non_negative"

        df_selected["combined_non_negative"] = np.maximum([x for x in list(df_selected["combined"])], 0).tolist()

    fig = px.scatter(df_selected, y="specificity", x="sensitivity", color=first_col,
                     size=chosen_size_param, hover_data=["combined", "gene"])
    return fig
# -------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False)
