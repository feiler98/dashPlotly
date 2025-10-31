# dependencies
from dash import Dash, html, dash_table, callback, Output, Input, dcc
import pandas as pd
from pathlib import Path
import plotly.express as px


# import data
path_tirosh_dir = Path("/home/feilerwe/bench_data/dataloader/other/marker_genes__tirosh_group")
dict_tirosh_data = {p.stem: pd.read_csv(p) for p in path_tirosh_dir.glob("*.csv")}
dict_keys = {tag.replace("_", " "): tag for tag in dict_tirosh_data.keys()}

app = Dash()
app.layout = [
    html.Img(src="/static/assets/weizmann_institute.png", alt="Weizmann Institute Banner", style={"width":"20vw", "min-width":"250px"}),
    html.Hr(),
    dcc.Dropdown(options=list(dict_keys.keys()), value=list(dict_keys.keys())[0], id="radio_tirosh_table"),
    html.Hr(),
    dash_table.DataTable(columns=[], data=[], id="input_tirosh_table", page_size=(), filter_action="native"),
    # records is essential so that the app runs
    dcc.Dropdown(options=["10 rows", "20 rows", "50 rows", "100 rows"], value="10 rows", id="radio_tirosh_len_table", style={"width": "120px", "height":"35px"}),
    html.Button(children="reset filter", id="clear_filter", n_clicks=0, style={"width": "120px", "height":"35px", "margin-top":"5px", "border-color":"#ad1a38", "border-width":"thin", "border-radius":"5px", "background":"#e64e6d"}),
    html.Hr(),
    dcc.Graph(figure={}, id="input_tirosh_figure", style={"width": "97vw", "height": "65vh", "min-height": "300px"}),
    html.Hr()
]


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
    Input(component_id="radio_tirosh_table", component_property="value")
)
def update_figure(chosen_table: str):
    key = dict_keys[chosen_table]
    df_selected = dict_tirosh_data[key].copy()
    df_selected["balanced accuracy"] = [round((sens + spec) / 2, 4) for sens, spec in
                                        zip(list(df_selected["sensitivity"]), list(df_selected["specificity"]))]
    first_col = list(df_selected.columns)[0]
    fig = px.scatter(df_selected, y="specificity", x="sensitivity", color=first_col,
                     size="balanced accuracy", hover_data=["gene"])
    return fig
# -------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False)
