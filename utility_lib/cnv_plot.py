# import
# ----------------------------------------------------------------------------------------------------------------------
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
from pyomics import utils as ut
from collections import Counter
import pandas as pd
import multiprocessing as mp
import os
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------

# helpful resources
# -----------------
# x-axis connected figures in dash plotly --> https://stackoverflow.com/questions/75871154/plotly-share-x-axis-for-subset-of-subplots
# periodic table of elements --> https://plotly.com/python/annotated-heatmap/

# data genomic data
general_data_path = Path(__file__).parent.parent / "data" / "genome"


# data preparation
# ----------------------------------------------------------------------------------------------------------------------
def list_transpose(l: list) -> list:
    return list(map(list, zip(*l)))


def convert_ucsc_cytoband(path: (str | Path)) -> pd.DataFrame:
    df_cytoband = pd.read_csv(Path(path), sep="\t", names=["CHR", "START", "END", "tag", "tag_alter"]).dropna(how="any", axis=0)
    set_chr = list(df_cytoband["CHR"].unique())
    list_concat = []
    for chr_tag in set_chr:
        slice_df = df_cytoband.where(df_cytoband["CHR"] == chr_tag).dropna().sort_values(by="START", ascending=True)
        slice_df["tag_arm"] = [list(t)[0] for t in list(slice_df["tag"])]
        slice_df.drop_duplicates(subset="tag_arm", keep="first", inplace=True)
        list_concat.append(slice_df)
    return pd.concat(list_concat).astype({"START":int, "END": int}).set_index("CHR").drop(["tag", "tag_alter", "END"], axis=1)


def map_arm_by_chr_pos(bin_df: pd.DataFrame, df_arms: pd.DataFrame):
    list_genomic_arm_tag = [f"chr-arm | {"p" if df_arms.where(df_arms["tag_arm"] == "q").dropna().loc[row["CHR"], "START"] > row["START"] else "q"}" for _, row in
                              bin_df.iterrows()]
    return list_genomic_arm_tag


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


def mult_get_bin_info(slice_df: pd.DataFrame, df_gene_loc: pd.DataFrame):
    info_genes_per_bin_list = [get_gene_string(df_gene_loc, row["CHR"], row["START"], row["END"]) for _, row in
                               slice_df.iterrows()]
    slice_df["genes_txt"] = info_genes_per_bin_list
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

    with mp.Pool(os.cpu_count()) as pool:
        df_slices_list = pool.starmap(mult_get_bin_info,
                                      [(df_chunk, df_gene_loc) for df_chunk in
                                       df_chunk_splitter(df_chr_bins, os.cpu_count() * 2)])
    df_concat = pd.concat(df_slices_list, axis=0).sort_index(ascending=True, axis=0)
    return df_concat.set_index("index"), tuple_match


# plotting
# ----------------------------------------------------------------------------------------------------------------------
def add_heatmap_tile(figure_obj: go.Figure, coordinates_xy: tuple, rgb: tuple, info_str: str, showlegend=False):
    figure_obj.add_trace(go.Scatter(x=coordinates_xy[0],  # top-right, bottom-right, bottom-left, top-left
                                    y=coordinates_xy[1],
                                    fill='toself',
                                    fillcolor=f"rgb{tuple(rgb)}",
                                    line={"width":0},
                                    marker=None,
                                    mode='lines',
                                    name=info_str,
                                    hoverinfo="text",
                                    showlegend=showlegend))


# correct one to use
def build_cnv_heatmap(path_cnv_csv: (str | Path), assembly_genome: str = "hg_38", zero_one_norm: bool = False):
    """
    Creates a plotly heatmap of infercnv data

    Parameters
    ----------
    path_cnv_csv: str | Path
    assembly_genome: str
    zero_one_norm: bool
    """

    # setting up the plotting dependencies
    # ------------------------------------
    # plot colors:
    parent_path = Path(path_cnv_csv).parent
    path_genome_data = Path(__file__).parent.parent / "data" / "genome"
    list_available_genome = [p.stem.split("__")[0] for p in path_genome_data.glob("*.gtf")]
    if assembly_genome not in list_available_genome:
        raise ValueError(f"Given genome {assembly_genome} is not available! Currently available {list_available_genome}.")

    # import and filter
    df_cnv = pd.read_csv(path_cnv_csv)
    df_arms = pd.read_csv(general_data_path / f"{assembly_genome}__loc_qp_arm.csv", index_col="CHR")
    allowed_chr_list = list(set(df_arms.index))
    df_cnv = df_cnv.where(df_cnv["CHR"].isin(allowed_chr_list)).dropna()

    slice_data_list = ["CHR", "START", "END"]
    col_data = [c for c in df_cnv.columns if c not in slice_data_list]
    # true values of CNV method
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

    if not zero_one_norm:
        min_val = 0
    # normalized dataframe --> 0-1 min-max
    df_norm = slice_data.map(lambda x: round((x - min_val) / (max_val - min_val), 2))

    # position of bins
    bin_df = df_cnv[slice_data_list]

    #########################
    # HOVER TEXT & Plotting #
    #########################
    list_genomic_pos_hm_tile = [f"{row["CHR"]} | {row["START"]} - {row["END"]}" for _, row in bin_df.iterrows()]

    # counting chromosome tag occurrences for second plot with bin regions --> CHR fig on top of CNA fig
    list_chr_tag_count = [tag.split(" |")[0].replace("chr", "Chromosome ") for tag in list_genomic_pos_hm_tile]
    count_dict_chr = dict(Counter(list_chr_tag_count))


    slice_pos, _ = calc_absolute_bin_position(bin_df, assembly_genome)
    list_genes_text = slice_pos["genes_txt"].tolist()  # list with all genes per bin
    list_genomic_arm_tag = map_arm_by_chr_pos(bin_df, df_arms)
    list_text = [[arm_tag]*len(col_data) for arm_tag in list_genomic_arm_tag]
    list_text_t = np.array(list_transpose(list_text))

    labels_dict = dict(
        y="cells",
        x="genomic_bins",
        color="normalized CNA-value"
    )

    # create a multiplot
    # ------------------
    fig_vstack = make_subplots(rows=2,
                               cols=1,
                               shared_xaxes=True,
                               vertical_spacing=0.02)

    # top plot describing chromosome positions and genes located at the respective bins
    # colors  --> alternating
    c1_chr = "#a448f0"  # chromosome total bin color
    c2_chr = "#413be3"
    b1_chr = "#ac60eb"  # genomic region bin within chromosome color
    b2_chr = "#5a55e6"

    colorscale = [[0, c1_chr],
                  [1, c2_chr],
                  [2, b1_chr],
                  [3, b2_chr]]

    bin_size = len(col_data)
    blank_list = [""]*bin_size

    # bottom bin (genes for each genomic region)
    bin_color_list = [(x%2) + 2 for x in range(1, bin_size+1, 1)]
    bin_hover_list = [f"genSeg {i+1}<br>{txt[0]}<br>{txt[1]}<br>{txt[2]}" for i, txt in enumerate(zip(list_genomic_pos_hm_tile, list_genomic_arm_tag, list_genes_text))]

    # top bin (chromosomes)
    list_chr_color = []
    list_chr_text = []
    i = 1
    for name_key, len_bin_chr_abs in count_dict_chr.items():
        list_color = [i%2]*len_bin_chr_abs
        list_chr_color.extend(list_color)

        sub_list_text = []
        len_mid_point = int(len_bin_chr_abs/2)
        len_rest = len_bin_chr_abs-(len_mid_point+1)
        sub_list_text.extend([""]*len_mid_point)
        sub_list_text.append(name_key)
        sub_list_text.extend([""] * len_rest)
        list_chr_text.extend(sub_list_text)

        i += 1

    top_hover_arr = np.array([blank_list,
                              bin_hover_list])
    top_txt_arr = np.array([list_chr_text,
                            blank_list])
    top_color_arr = np.array([list_chr_color,
                              bin_color_list])

    # top plot
    # --------
    top_fig = ff.create_annotated_heatmap(top_color_arr[::-1],
                                          annotation_text=top_txt_arr[::-1],
                                          text=top_hover_arr[::-1],
                                          colorscale=colorscale,
                                          font_colors=['black'],
                                          hoverinfo='text')
    fig_vstack.add_trace(top_fig,
                         row=1,
                         col=1)

    # main CNA plot
    # -------------
    cna_fig = px.imshow(df_norm.T.to_numpy(),
                    y=list(df_norm.columns),
                    x=list_genomic_pos_hm_tile,
                    labels = labels_dict,
                    text_auto=False,
                    aspect="auto",
                    title=f"InferCNA plot | {path_cnv_csv.stem}",
                    color_continuous_scale=["#303bd9", "#ffffff", "#f258fc"])
    cna_fig.update_layout(title_font_size=24)
    cna_fig.update_xaxes(showticklabels=False)
    cna_fig.update(data=[{"customdata": list_text_t,
                      "hovertemplate":"CELL || %{y} <br>val-CNA | %{z}% <br>%{x} <br>%{customdata}"}])

    fig_vstack.add_trace(cna_fig,
                         row=2,
                         col=1)

    plotly.offline.plot(fig_vstack, filename=str(parent_path / f"plot__{path_cnv_csv.stem}.html"))



# docker testing
if __name__ == "__main__":
    # fix CHR columns with lambda
    # df_cnv = pd.read_csv(path_ck_csv)
    # df_cnv["CHR"] = df_cnv["CHR"].map(lambda x: f"chr{x}")
    path_ck_cnv = Path(__file__).parent.parent / "data" / "test_cnv_plot"
    build_cnv_heatmap(path_ck_cnv / "copykat_curated.csv")