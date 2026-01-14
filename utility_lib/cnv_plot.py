# import
# ----------------------------------------------------------------------------------------------------------------------
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
# subplots --> https://plotly.com/python/table-subplots/
# subplots api --> https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html
# layout settings --> https://plotly.com/python/reference/layout/
# colorbar settings --> https://plotly.com/python/reference/#heatmap-colorbar

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


def calc_pred_saturation(df_cna_idx, assembly_genome: str = "hg_38"):
    slice_data_list = ["CHR", "START", "END"]
    available_cols = list(df_cna_idx.columns)

    # check input
    for col_tags in slice_data_list:
        if not col_tags in available_cols :
            raise ValueError(f"{slice_data_list} must be contained in DataFrame columns! Only {available_cols} were found.")
    df_cna_idx["DIFF"] = df_cna_idx["END"] - df_cna_idx["START"]
    df_cna_idx_unique_chr_list = list(df_cna_idx["CHR"].unique())
    df_chr_total_len = pd.read_csv(Path(general_data_path / f"{assembly_genome}__chr_bin_lengths__ucsc.csv"), index_col="CHR")
    dict_df = {"":["pred<br>saturation"]}
    for idx, row in df_chr_total_len.iterrows():
        if not idx in df_cna_idx_unique_chr_list:
            continue
        chr_slice_sum_diff = sum(list(df_cna_idx.where(df_cna_idx["CHR"] == idx).dropna()["DIFF"]))
        chr_diff_percent = round((chr_slice_sum_diff/row["bin_size"])*100, 2)
        dict_df[f"<br>{str(idx)}"] = [f"<br>{chr_diff_percent} %"]

    total_diff_percent = round((sum(list(df_cna_idx["DIFF"]))/sum(list(df_chr_total_len["bin_size"])))*100, 2)
    dict_df["total<br>Genome"] = [f"<br>{total_diff_percent} %"]

    return pd.DataFrame.from_dict(dict_df)

# plotting
# ----------------------------------------------------------------------------------------------------------------------

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
    print("> import and preparation of data [✓]")
    allowed_chr_list = list(set(df_arms.index))
    df_cnv = df_cnv.where(df_cnv["CHR"].isin(allowed_chr_list)).dropna()
    print("> filtering of valid chromosomal-sections [✓]")

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
    print("> normalization of data [✓]")
    # position of bins
    bin_df = df_cnv[slice_data_list]

    #########################
    # HOVER TEXT & Plotting #
    #########################

    # plotting element sizes  [px]
    # ----------------------------
    table_height = 150
    main_cna_height = len(col_data)
    sum_cna_height = 30
    gene_height = 40
    chr_bin_height = 40

    vstack_height = table_height+main_cna_height+sum_cna_height+gene_height+chr_bin_height

    list_genomic_pos_hm_tile = [f"{row["CHR"]} | {int(row["START"])} - {int(row["END"])}" for _, row in bin_df.iterrows()]

    slice_pos, _ = calc_absolute_bin_position(bin_df, assembly_genome)

    list_genes_text = slice_pos["genes_txt"].tolist()  # list with all genes per bin
    list_genomic_arm_tag = map_arm_by_chr_pos(bin_df, df_arms)
    list_text = [[arm_tag]*len(col_data) for arm_tag in list_genomic_arm_tag]
    list_text_t = np.array(list_transpose(list_text))
    print("> mapping of genes located within genomic bins [✓]")

    # counting chromosome tag occurrences for second plot with bin regions --> CHR fig on top of CNA fig
    list_chr_tag_count = [tag.split(" |")[0].replace("chr", "Chromosome ") for tag in list_genomic_pos_hm_tile]
    count_dict_chr = dict(Counter(list_chr_tag_count))  # is in order -> len list == length of square

    list_chr_info = []
    list_chr_val = []

    for i, key, val in zip(range(0, len(count_dict_chr), 1), list(count_dict_chr.keys()),
                           list(count_dict_chr.values())):
        list_chr_info.extend([key] * val)
        list_chr_val.extend([(i % 2) * -0.5 - 0.5] * val)
    list_chr_info_update = [f"{chr_tag}<br>   {arm}" for chr_tag, arm in zip(list_chr_info, list_genomic_arm_tag)]
    print("> mapping of chromosomal bins [✓]")

    # create table underneath plot
    df_summary = calc_pred_saturation(bin_df, assembly_genome)
    print("> generation of summary table [✓]")

    # create a multiplot
    # ------------------
    fig_vstack = make_subplots(rows=5,
                               cols=1,
                               shared_xaxes=True,
                               vertical_spacing=0.02,
                               row_heights=[chr_bin_height/vstack_height,
                                            gene_height/vstack_height,
                                            sum_cna_height/vstack_height,
                                            main_cna_height/vstack_height,
                                            table_height/vstack_height],
                               specs=[[{"type": "xy"}],
                                      [{"type": "xy"}],
                                      [{"type": "xy"}],
                                      [{"type": "xy"}],
                                      [{"type": "table"}]])

    # top plot describing chromosome positions and genes located at the respective bins
    # colors --> alternating

    bin_size = len(df_norm)

    # bottom bin (genes for each genomic region)
    bin_color_list = [(x%2)*0.1 + 0.5 for x in range(1, bin_size+1, 1)]
    bin_hover_list = [f"GenoSeg | {i+1}<br>{txt[0]}<br>{txt[1]}<br>genes:<br>   {txt[2]}" for i, txt in enumerate(zip(list_genomic_pos_hm_tile, list_genomic_arm_tag, list_genes_text))]
    print("> generation of hover text [✓]")

    # chromosome plot
    # ---------------
    top_fig = px.imshow([list_chr_val],
                        y=["Chromosomes"],
                        x=list_genomic_pos_hm_tile,
                        text_auto=False,
                        aspect="auto")
    top_fig.update(data=[{"customdata": [list_chr_info_update],
                          "hovertemplate":"%{customdata} <extra></extra>"}])

    fig_vstack.add_trace(top_fig.data[0],
                         row=1,
                         col=1)
    # gene plot
    # ---------
    middle_fig = px.imshow([bin_color_list],
                            y=["Genes"],
                            x=list_genomic_pos_hm_tile,
                            text_auto=False,
                            aspect="auto")
    middle_fig.update(data=[{"customdata": [bin_hover_list],
                          "hovertemplate":"%{customdata} <extra></extra>"}])

    fig_vstack.add_trace(middle_fig.data[0],
                         row=2,
                         col=1)

    # main-sum CNA plot
    # -----------------
    cna_fig = px.imshow([df_norm.T.mean(axis=0).to_numpy()],
                    y=["summarized CNA"],
                    x=list_genomic_pos_hm_tile,
                    text_auto=False,
                    aspect="auto")
    cna_fig.update(data=[{"customdata": [list_text_t[0]],
                          "hovertemplate":"Cell | %{y} <br>   val-CNA | %{z} <br>   %{x} <br>   %{customdata} <extra></extra>"}])

    fig_vstack.add_trace(cna_fig.data[0],
                         row=3,
                         col=1)

    # main CNA plot
    # -------------
    cna_fig = px.imshow(df_norm.T.to_numpy(),
                    y=list(df_norm.columns),
                    x=list_genomic_pos_hm_tile,
                    text_auto=False,
                    aspect="auto")
    cna_fig.update(data=[{"customdata": list_text_t,
                          "hovertemplate":"Cell | %{y} <br>   val-CNA | %{z} <br>   %{x} <br>   %{customdata} <extra></extra>"}])

    fig_vstack.add_trace(cna_fig.data[0],
                         row=4,
                         col=1)

    # table
    # -----
    table_summary = go.Table(
        header=dict(
            values=list(df_summary.columns),
            fill=dict(color="#c9365b"),
            font=dict(size=15, color="white", weight="bold"),
            align="center"
        ),
        cells=dict(
            values=[df_summary[c].tolist() for c in df_summary.columns],
            fill=dict(color="#e05175"),
            font=dict(size=12, color="white"),
            align="center")
    )

    fig_vstack.add_trace(table_summary,
                          row=5,
                          col=1)

    # settings of fig_vstack
    fig_vstack.update_xaxes(showticklabels=False)
    fig_vstack.update_layout(coloraxis=dict(colorscale=[[0, "#303bd9"], [0.5, "#ffffff"], [1.0, "#f27933"]],
                                            colorbar=dict(len=int(main_cna_height*0.7),
                                                          lenmode="pixels")),
                             height=vstack_height,
                             showlegend=False,
                             title_font_weight="bold",
                             title_font_size=38,
                             title_text=f"InferCNA plot | {path_cnv_csv.stem}")

    print("> generation of html document [✓]")

    # plot, show, and save figure
    # ---------------------------
    plotly.offline.plot(fig_vstack , filename=str(parent_path / f"plot__{path_cnv_csv.stem}.html"))



# docker testing
if __name__ == "__main__":
    # fix CHR columns with lambda
    # df_cnv = pd.read_csv(path_ck_csv)
    # df_cnv["CHR"] = df_cnv["CHR"].map(lambda x: f"chr{x}")
    path_ck_cnv = Path(__file__).parent.parent / "data" / "test_cnv_plot"
    build_cnv_heatmap(path_ck_cnv / "copykat_curated.csv")