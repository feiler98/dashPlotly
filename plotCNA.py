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
import numpy as np

# local imports
from utility_lib import (calc_absolute_bin_position,
                         map_arm_by_chr_pos,
                         list_transpose,
                         calc_pred_saturation)
# ----------------------------------------------------------------------------------------------------------------------

# helpful resources
# -----------------
# x-axis connected figures in dash plotly --> https://stackoverflow.com/questions/75871154/plotly-share-x-axis-for-subset-of-subplots
# periodic table of elements --> https://plotly.com/python/annotated-heatmap/
# subplots --> https://plotly.com/python/table-subplots/
# subplots api --> https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html
# layout settings --> https://plotly.com/python/reference/layout/
# colorbar settings --> https://plotly.com/python/reference/#heatmap-colorbar
# two coloraxis in subplots --> https://community.plotly.com/t/subplots-of-two-heatmaps-overlapping-text-colourbar/38587/9


# data genomic data
# -----------------
general_data_path = Path(__file__).parent / "data" / "genome"


def build_cnv_heatmap(df_cnv: pd.DataFrame,
                      path_out: (str | Path) = Path.cwd(),
                      data_title: str = None,
                      df_cellclass: (pd.DataFrame | None)  = None,
                      df_cellclass_classify_by: (str | None) = None,
                      sort: str = False,
                      df_cellclass_filter_col: bool = False,
                      assembly_genome: str = "hg_38",
                      zero_one_norm: bool = False):
    """
    Creates an express plotly imshow heatmap of infercnv data.

    Parameters
    ----------
    df_cnv: pd.DataFrame
        DataFrame CNA-matrix (usually from .csv); output of any InferCNV-like method.
        [index --> chr-bins; columns --> 'CHR', 'START', 'END', cells]
    path_out: str | Path
        Location where .html file saved;
    data_title: str
    df_cellclass: pd.DataFrame | None
        DataFrame cell classification; only overlap cells (as index) are used; if no overlap --> ValueError
    df_cellclass_classify_by: str | None
        None -> applies order as received; takes first column for CNA-mean calculation
        str -> column name of DataFrame.
    sort: str
        Sorts df_cellclass by target column if 'ascending' or 'descending' tags are provided, otherwise disabled.
    df_cellclass_filter_col: bool
        Removes all columns that are not used for the primary classification if True.
    assembly_genome: str
        Genome the original UMI-matrix / BAM-file was mapped to; available: 'hg_19', 'hg_38'.
    zero_one_norm: bool
        If True, range 0 to 1, else -1 to 1
    """

    # setting up the plotting dependencies
    # ------------------------------------
    list_available_genome = [p.stem.split("__")[0] for p in general_data_path.glob("*.gtf")]
    if assembly_genome not in list_available_genome:
        raise ValueError(f"Given genome {assembly_genome} is not available! Currently available {list_available_genome}.")
    df_arms = pd.read_csv(general_data_path / f"{assembly_genome}__loc_qp_arm.csv", index_col="CHR")
    print("> preparation of cnv data [✓]")
    allowed_chr_list = list(set(df_arms.index))
    df_cnv = df_cnv.where(df_cnv["CHR"].isin(allowed_chr_list)).dropna()
    print("> filtering of valid chromosomal-sections [✓]")

    if isinstance(df_cellclass, pd.DataFrame) or df_cellclass is not None:
        raise ValueError(f"Argument 'df_cellclass' expected (None | pd.DataFrame) as input, received '{type(df_cellclass)}' instead!")

    # split the CNV DataFrame into its 2 sections --> multi-index and cells
    slice_data_list = ["CHR", "START", "END"]
    col_data = [c for c in df_cnv.columns if c not in slice_data_list]

    # additional data in form of cell classes (col 1 of 2)
    if df_cellclass is not None:
        idx_cellclass = list(df_cellclass.index)
        union_cells = [c_tag for c_tag in idx_cellclass if c_tag in col_data]
        if not len(union_cells) > 0:
            raise ValueError("Provided DataFrame 'df_cellclass' must have a union with the CNV-matrix > 0!")

        # if union > 0 --> use union_cells as col_data list
        df_cellclass = df_cellclass.loc[union_cells, ::]
        flag_sort = False
        if df_cellclass_classify_by is not None:
            if not str(df_cellclass_classify_by) in df_cellclass.columns:
                raise ValueError(f"Column '{df_cellclass_classify_by}' is not in DataFrame 'df_cellclass'!")
            if df_cellclass_filter_col:
                df_cellclass = df_cellclass[df_cellclass_classify_by]
            # sort if passed
            dict_sort = {"ascending": True, "descending": False}
            if str(sort) in dict_sort.keys():
                df_cellclass = df_cellclass.sort_values(df_cellclass_classify_by,
                                                        ascending=dict_sort[str(sort)],
                                                        axis=0)
                flag_sort = True
        col_data = list(df_cellclass.index)
        print(f"> Matching, {"filtering, and sorting" if flag_sort else "and filtering"} of cell-classification [✓]")

    # true values of CNV method
    slice_data = df_cnv[col_data]

    # min-max normalization of the data
    # ---------------------------------
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
    chr_bin_height = 70
    gene_height = 35
    sum_cna_height = 35
    main_cna_height = len(col_data)
    table_height = 400
    chr_length = 600


    vstack_height = table_height+main_cna_height+sum_cna_height+gene_height+chr_bin_height+chr_length

    list_genomic_pos_hm_tile = [f"{row["CHR"]} | {int(row["START"])} - {int(row["END"])}" for _, row in
                                bin_df.iterrows()]

    slice_pos, _ = calc_absolute_bin_position(bin_df, assembly_genome)

    list_genes_text = slice_pos["genes_txt"].tolist()  # list with all genes per bin
    list_genomic_arm_tag = map_arm_by_chr_pos(bin_df, df_arms)
    list_text = [[arm_tag]*len(col_data) for arm_tag in list_genomic_arm_tag]
    list_text_t = np.array(list_transpose(list_text))
    print("> mapping of genes located within genomic bins [✓]")

    # counting chromosome tag occurrences for second plot with bin regions --> CHR fig on top of CNA fig
    list_chr_tag_count = [tag.split(" |")[0].replace("chr", "Chromosome ") for tag in
                          list_genomic_pos_hm_tile]
    count_dict_chr = dict(Counter(list_chr_tag_count))  # is in order -> len list == length of square

    list_chr_info = []
    list_chr_val = []

    for i, key, val in zip(range(0, len(count_dict_chr), 1), list(count_dict_chr.keys()),
                           list(count_dict_chr.values())):
        list_chr_info.extend([key] * val)
        list_chr_val.extend([((i+1) % 2) * -0.3 - 0.6] * val)
    list_chr_info_update = [f"{chr_tag}<br>   {arm}" for chr_tag, arm in zip(list_chr_info, list_genomic_arm_tag)]
    dict_chr_arm_encode = {"chr-arm | p": -0.3,"chr-arm | q": -0.1}
    list_chr_arm_val = list(map(dict_chr_arm_encode.get, list_genomic_arm_tag))

    print("> mapping of chromosomal bins [✓]")

    # create table underneath plot
    df_summary = calc_pred_saturation(bin_df, assembly_genome)
    print("> generation of summary table [✓]")

    # create a multiplot
    # ------------------
    # std settings
    col_pos = 1
    col_width = [1]

    # change col layout if additional data
    if df_cellclass is not None:
        col_pos = 2
        col_width = [2]

    fig_vstack = make_subplots(rows=6,
                               cols=col_pos,
                               shared_xaxes=True,
                               vertical_spacing=0.02,
                               row_heights=[chr_bin_height/vstack_height,
                                            gene_height/vstack_height,
                                            sum_cna_height/vstack_height,
                                            main_cna_height/vstack_height,
                                            table_height/vstack_height,
                                            chr_length/vstack_height],
                               column_widths=col_width,
                               specs=[[{"type": "xy"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "table"}]*col_pos,
                                      [{"type": "bar"}]*col_pos]
                               )

    # adjust width if additional data
    if df_cellclass is not None:
        fig_vstack.update_layout(col_width=[0.1, 0.9])

    # top plot describing chromosome positions and genes located at the respective bins
    # colors --> alternating

    bin_size = len(df_norm)

    # bottom bin (genes for each genomic region)
    bin_color_list = [(x%2)*0.1 + 0.5 for x in range(1, bin_size+1, 1)]
    bin_hover_list = [f"GenoSeg | {i+1}<br>{txt[0]}<br>{txt[1]}<br>genes:<br>   {txt[2]}" for i, txt in
                      enumerate(zip(list_genomic_pos_hm_tile, list_genomic_arm_tag, list_genes_text))]
    print("> generation of hover text [✓]")

    # chromosome plot
    # ---------------
    top_fig = px.imshow([list_chr_val, list_chr_arm_val][::-1],
                        y=["Chromosomes", "Chrom. arm"][::-1],
                        x=list_genomic_pos_hm_tile,
                        text_auto=False,
                        aspect="auto")
    top_fig.update(data=[{"customdata": [list_chr_info_update, list_chr_info_update],
                          "hovertemplate":"%{customdata} <extra></extra>"}])

    fig_vstack.add_trace(top_fig.data[0],
                         row=1,
                         col=col_pos)
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
                         col=col_pos)

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
                         col=col_pos)

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
                         col=col_pos)

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
                          col=col_pos)


    # chromosomal barplot
    # -------------------

    chr_multi_bar = [go.Bar(x=["A", "B", "C", "D"], y=[-100, -70, -100, -70], marker_color='rgb(69, 73, 135)', marker_line_color='rgba(0,0,0,0)', marker=dict(cornerradius="100%")),
                     go.Bar(x=["A", "B", "C", "D"], y=[-30, -50, -100, -70], marker_color='rgb(242, 48, 129)', marker_line_color='rgba(0,0,0,0)', marker=dict(cornerradius="100%")),
                     go.Bar(x=["A", "B", "C", "D"], y=[65, 50, 60, 30], marker_color='rgb(69, 73, 135)', marker_line_color='rgba(0,0,0,0)', marker=dict(cornerradius="100%")),
                     go.Bar(x=["A", "B", "C", "D"], y=[20, 10, 0, 20], marker_color='rgb(53, 185, 242)', marker_line_color='rgba(0,0,0,0)', marker=dict(cornerradius="100%"))]

    fig_vstack.add_trace(chr_multi_bar,
                          row=6,
                          col=col_pos)

    # settings of fig_vstack
    fig_vstack.update_xaxes(showticklabels=False)
    fig_vstack.update_layout(coloraxis=dict(colorscale=[[0, "#303bd9"], [0.5, "#ffffff"], [1.0, "#f27933"]],
                                            colorbar=dict(len=int(main_cna_height*0.7),
                                                          lenmode="pixels")),
                             height=vstack_height,
                             showlegend=False,
                             title_font_weight="bold",
                             title_font_size=38,
                             title_text=f"InferCNA plot {f'| {data_title}' if data_title is not None else ""}",
                             plot_bgcolor='rgb(34, 37, 87)',
                             barmode="overlay", yaxis6=dict(showgrid=False, showticklabels=False))

    print("> generation of html document [✓]")

    # plot, show, and save figure
    # ---------------------------
    plotly.offline.plot(fig_vstack , filename=str(path_out / f"plot__{str(data_title) if data_title is not None else "infercnv"}.html"))



# docker testing
if __name__ == "__main__":
    path_ck_cnv = Path(__file__).parent / "data" / "test_cnv_plot"
    build_cnv_heatmap(pd.read_csv(path_ck_cnv / "copykat_curated.csv"))