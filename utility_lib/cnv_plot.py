# import
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from pathlib import Path
from pyomics import utils as ut
from itertools import islice
import pandas as pd
import multiprocessing as mp
import os


general_data_path = Path(__file__).parent.parent / "data" / "genome"

# data preparation
# ----------------------------------------------------------------------------------------------------------------------

def get_gene_loc(assembly_genome: str = "hg_38") -> pd.DataFrame:
    df_genes = pd.read_csv(general_data_path / f"{assembly_genome}__refGene.gtf", sep="\t",
                           names=["CHR", "refGene", "type", "START", "END", '.', '+', '..1', "gene"]).drop(
        ['.', '+', '..1', "refGene", "type"], axis=1)
    df_genes["gene"] = df_genes["gene"].map(lambda x: x.split('"')[1])
    df_genes.reset_index(inplace=True, drop=True)
    return df_genes.astype({"START": int, "END": int})


def get_gene_string(df_get_gene_loc: pd.DataFrame, chr_tag: str, start_pos: int, end_pos: int) -> str:
    df_slice = df_get_gene_loc.where((df_get_gene_loc["CHR"] == chr_tag) & (df_get_gene_loc["START"] >= start_pos) & (
                df_get_gene_loc["END"] <= end_pos)).dropna()
    return "<br>   ".join(list(set(df_slice["gene"])))


def df_chunk_splitter(df: pd.DataFrame, chunk_amount: int) -> list:
    len_df = len(df)
    if chunk_amount <= 0:
        raise ValueError("chunk amount must be greater than 0!")
    if len_df < chunk_amount:
        chunk_amount = len_df
    chunk_size = int(len_df / chunk_amount)
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]


def mult_get_bin_info(slice_df: pd.DataFrame, tuple_bins: tuple, df_len_bins: pd.DataFrame, df_gene_loc: pd.DataFrame):
    info_genes_per_bin_list = [get_gene_string(df_gene_loc, row["CHR"], row["START"], row["END"]) for _, row in
                               slice_df.iterrows()]
    slice_df["genes_txt"] = info_genes_per_bin_list
    start_abs_list = [sum(tuple_bins[:df_len_bins.loc[row["CHR"], "index"]]) + row["START"] for _, row in
                      slice_df.iterrows()]
    slice_df["STARTabs"] = start_abs_list
    slice_df["ENDabs"] = slice_df["STARTabs"] + slice_df["diff"]
    return slice_df


def calc_absolute_bin_position(df_chr_bins: pd.DataFrame, assembly_genome: str = "hg_38"):
    # get genes & positions
    df_gene_loc = get_gene_loc(assembly_genome)

    # import df with chrom bins
    path_genomic_bins = Path(general_data_path / f"{assembly_genome}__chr_bin_lengths__ucsc.csv")
    df_len_bins = pd.read_csv(path_genomic_bins).set_index("CHR")
    df_len_bins["index"] = range(0, len(df_len_bins))
    # filter chr bins
    list_chr_available = list(set(df_chr_bins.loc[:, "CHR"]))
    list_chr_allowed = list(df_len_bins.index)
    tuple_match = [c for c in list_chr_allowed if c in list_chr_available]
    df_chr_bins = df_chr_bins.reset_index()
    df_chr_bins = df_chr_bins[df_chr_bins.loc[:, "CHR"].isin(tuple_match)]  # throw out all chromosomes which do not fit!
    df_chr_bins["diff"] = df_chr_bins.loc[:, "END"] - df_chr_bins.loc[:, "START"]

    tuple_bins = tuple(df_len_bins["bin_size"])
    with mp.Pool(os.cpu_count()) as pool:
        df_slices_list = pool.starmap(mult_get_bin_info,
                                      [(df_chunk, tuple_bins, df_len_bins, df_gene_loc) for df_chunk in
                                       df_chunk_splitter(df_chr_bins, os.cpu_count() * 2)])
    df_concat = pd.concat(df_slices_list, axis=0).sort_values("STARTabs", ascending=True)
    return df_concat.set_index("index"), tuple_match # better safe than sorry for matching the correct index


def gen_cnv_abs_data(path_cnv_csv: (str, Path), assembly_genome: str = "hg_38") -> dict:
    parent_path = Path(path_cnv_csv).parent
    df_cnv = pd.read_csv(path_cnv_csv)
    slice_data_list = ["CHR", "START", "END"]
    col_data = [c for c in df_cnv.columns if c not in slice_data_list]
    slice_data = df_cnv[col_data]

    # min-max normalization of the data
    max_val = abs(slice_data.max().max())
    min_val = abs(slice_data.min().min())
    # centering 0
    if max_val > min_val:
        min_val = -max_val
    else:
        max_val = min_val
        min_val = -min_val
    # normalized dataframe --> 0 is centered!
    df_norm = slice_data.map(lambda x: (x - min_val) / (max_val - min_val))
    df_norm.drop("Unnamed: 0", axis=1)
    # absolute position of genomic bins
    slice_pos, tuple_match = calc_absolute_bin_position(df_cnv[slice_data_list], assembly_genome)
    df_norm_abs_cnv = pd.concat([slice_pos, df_norm], axis=1)

    # transform to a dictionary
    # --> JSON as export format / input for the graph generation

    # construction of file
    cnv_plot_json_dict = {}
    """
    JSON structure
    ##############
    
    main keys:
    - chr  --> dictionary with chromosome box coordinates + color + info text
    - cells  --> dictionary with heatmap tiles of chromosome based on df_norm_abs_cnv
    
    chr  --> subkeys
    ----------------
    coordinates_xy --> ([top-r, bot-r, bot-l, top-l], [top-r, bot-r, bot-l, top-l])
    rgb --> (r, g, b)
    info_str --> chr-tag
    
    
    cells  --> subkeys
    ------------------
    coordinates_xy --> ([top-r, bot-r, bot-l, top-l], [top-r, bot-r, bot-l, top-l])
    rgb --> (r, g, b)
    info_str --> cell_name
                 chrX:start-stop
                 Signal
                 Genes:
                    gene_1
                    ...
                    gene_n
    """

    # generating the chromosomal-bins
    # -------------------------------
    path_genomic_bins = Path(general_data_path / f"{assembly_genome}__chr_bin_lengths__ucsc.csv")
    df_len_bins = pd.read_csv(path_genomic_bins).set_index("CHR")
    df_len_bins = df_len_bins.loc[tuple_match,:]
    tuple_bins = tuple(df_len_bins["bin_size"])
    chr_collect_dict = {}
    for i, len_chr in enumerate(tuple_bins):
        abs_start_pos = sum(tuple_bins[:i])
        abs_end_pos = sum(tuple_bins[:(i + 1)])
        top = len(df_norm.columns)+0.5
        bottom = 0
        # alternating colors for chromosomal bins
        if i % 2 == 0:
            rgb_tuple = (212, 212, 212)
        else:
            rgb_tuple = (114, 117, 138)
        tag_chr = tuple_match[i]
        dict_chr = {tag_chr: {"coordinates_xy":([abs_end_pos, abs_end_pos, abs_start_pos, abs_start_pos], [top, bottom, bottom, top]),
                              "rgb":rgb_tuple,
                              "info_str":f"Chromosome {tag_chr.replace("chr", "")}<br>Total length: {len_chr}"}}
        chr_collect_dict.update(dict_chr)
    # update json dict
    cnv_plot_json_dict.update({"chr": chr_collect_dict})

    # generating the heatmap tiles
    # ----------------------------
    list_cells = list(df_norm.columns)
    mult_rows_list = [row for _, row in df_norm_abs_cnv.iterrows()]
    with mp.Pool(os.cpu_count()) as pool:
        list_of_dicts = pool.starmap(process_df_row, [(row, list_cells) for row in mult_rows_list])
    dict_cell_tiles = {}
    for sub_dicts in list_of_dicts:
        dict_cell_tiles.update(sub_dicts)
    # update json dict
    cnv_plot_json_dict.update({"cells": dict_cell_tiles})
    cnv_plot_json_dict.update({"cell_tags": list_cells})
    map_cell_pos_yaxis = [x+0.5 for x in range(0, len(list_cells), 1)]
    cnv_plot_json_dict.update({"cells_y_pos": map_cell_pos_yaxis})

    ut.save_as_json_dict(cnv_plot_json_dict, parent_path, f"{path_cnv_csv.stem}__dash_cnv_matrix")
    return cnv_plot_json_dict


def process_df_row(df_row: pd.Series, list_cells: list) -> dict:
    dict_append = {}
    abs_start_pos = df_row["STARTabs"]
    abs_end_pos =  df_row["ENDabs"]

    for y_pos, cell_id in enumerate(list_cells):
        norm_val = df_row[cell_id]
        export_tag = f"{cell_id}:{abs_start_pos}-{abs_end_pos}"
        top = y_pos+1
        bottom = y_pos
        if df_row["genes_txt"] is not None:
            genes_text = "   " + str(df_row["genes_txt"])
        else:
            genes_text = "   no gene information available"
        dict_cell = {export_tag: {"coordinates_xy": ([abs_end_pos, abs_end_pos, abs_start_pos, abs_start_pos],
                                                    [top, bottom, bottom, top]),
                                 "rgb": bwr_color(norm_val),
                                 "info_str": f"Cell {cell_id}<br>{df_row["CHR"]}: {df_row["START"]}-{df_row["END"]}<br>Genes:<br>{genes_text}"}}
        dict_append.update(dict_cell)
    return dict_append


# plotting
# ----------------------------------------------------------------------------------------------------------------------

# functions for constructing the plot
def bwr_color(value: int) -> tuple:
    """
    Parameters
    ----------
    value: int
        Expects a 0-1 normalized value.
        x <= 0 results in a blue color.
        0.5 results in white.
        x >= 1 results in red

    Returns
    -------

    """
    value = int(value * 255)  # must be int, otherwise value will be red
    cmap = mpl.colormaps.get_cmap('bwr')
    rgb_tuple = tuple(map(lambda x: int(x * 255), cmap(value)))[:3]  # split off the alpha
    return rgb_tuple


def add_heatmap_tile(go_figure_obj: go.Figure, coordinates_xy: tuple, rgb: tuple, info_str: str, showlegend=False):
    go_figure_obj.add_trace(go.Scatter(x=coordinates_xy[0],  # top-right, bottom-right, bottom-left, top-left
                                       y=coordinates_xy[1],
                                       fill='toself',
                                       fillcolor=f"rgb{tuple(rgb)}",
                                       line={"width":0},
                                       marker=None,
                                       mode='lines',
                                       name=info_str,
                                       showlegend=showlegend))


# multiprocessing function for add_heatmap_tile
def mult_run_tile_generator(json_dict_slice, go_figure_obj):
    for values in json_dict_slice.value():
        add_heatmap_tile(go_figure_obj, **values)


def dict_to_chunks(data_dict, chunk_size=10000):
    it = iter(data_dict)
    for i in range(0, len(data_dict), chunk_size):
        yield {k:data_dict[k] for k in islice(it, chunk_size)}


def build_cnv_plot(path_json: (str, Path), header: (str, None) = None):
    """
    Parameters
    ----------
    path_json: str | Path
    header: str | None
    """
    path_json = Path(path_json)
    path_out = path_json.parent

    # import HSON file
    json_dict = ut.get_json_dict(path_json)

    # figure construction
    # -------------------
    if header is None:
        header = path_json.stem.split("__")[0]
    fig = go.Figure(go.Layout(title=go.Layout.Title(text=f"CNV prediction | {header}")))
    fig.update_layout(xaxis_title="cell entities",
                      yaxis_title="unified chromosomal position",
                      yaxis= dict(
                          tickmode = "array",
                          tickvals = json_dict["cells_y_pos"],
                          ticktext = json_dict["cell_tags"]
                      ))
    # chromosomes
    for sub_dict_val in json_dict["chr"].values():
        add_heatmap_tile(fig, showlegend=False, **sub_dict_val)

    # heatmap tiles
    """
    gen_chunk_cells = dict_to_chunks(json_dict["cells"])
    list_data_packages = [(slice_cells, fig) for slice_cells in gen_chunk_cells]
    with mp.Pool(os.cpu_count()) as pool:
        pool.starmap(mult_run_tile_generator, list_data_packages)
    """
    # export as html for easy and fast access
    fig.write_html(path_out / f"{header}.html")  # we have a sorting issue with the algorithm



# docker testing
if __name__ == "__main__":
    # fix CHR columns with lambda
    # df_cnv = pd.read_csv(path_ck_csv)
    # df_cnv["CHR"] = df_cnv["CHR"].map(lambda x: f"chr{x}")
    path_ck_cnv = Path(__file__).parent.parent / "data" / "test_cnv_plot" / "copykat_curated.csv"
    gen_cnv_abs_data(path_ck_cnv)
    build_cnv_plot(path_ck_cnv.parent / "copykat_curated__dash_cnv_matrix.json")
    build_cnv_plot(path_ck_cnv.parent / "copykat_curated__dash_cnv_matrix.json")