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
from utility_lib import (df_cna_idx_get_gene_info,
                         map_arm_by_chr_pos,
                         sort_df_row_by_similarity,
                         calc_pred_saturation,
                         list_transpose,
                         encode_df_name_cols)
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
# subplots coloraxis (more promising) --> https://community.plotly.com/t/colorbar-for-each-facet-col/73456


# data genomic data
# -----------------
general_data_path = Path(__file__).parent / "data" / "genome"


def build_cnv_heatmap(df_cnv: pd.DataFrame,
                      path_out: (str | Path) = Path.cwd(),
                      data_title: str = None,
                      df_cellclass: (pd.DataFrame | None)  = None,
                      df_cellclass_classify_by: (str | None) = None,
                      sort: bool = False,
                      df_cellclass_filter_col: bool = False,
                      assembly_genome: str = "hg_38"):
    """
    Creates an express plotly imshow heatmap of infercnv data.
    Advanced visualization with custom data embedding and prediction saturation statistics

    Parameters
    ----------
    df_cnv: pd.DataFrame
        DataFrame CNA-matrix (usually from .csv); output of any InferCNV-like method.
        [index --> chr-bins; columns --> 'CHR', 'START', 'END', *cells]
    path_out: str | Path
        Location where .html file saved;
    data_title: str
    df_cellclass: pd.DataFrame | None
        DataFrame cell classification; only overlap cells (as index) are used; if no overlap --> ValueError
    df_cellclass_classify_by: str | None
        None -> applies order as received; takes first column for CNA-mean calculation
        str -> column name of DataFrame.
    sort: bool
        If 'True' sorts the rows per section descending; most dissimilar array to 0 array on top,
        increasing dissimilarity between top row to bottom rows.
    df_cellclass_filter_col: bool
        Removes all columns that are not used for the primary classification if True.
    assembly_genome: str
        Genome the original UMI-matrix / BAM-file was mapped to; available: 'hg_19', 'hg_38'.
    """

    # initializing the function
    print_init = f"Construction of heatmap{f' < {data_title} >' if data_title is not None else ""}"
    print(f"""
{print_init}""")
    print("-"*len(print_init))

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

    if not isinstance(df_cellclass, pd.DataFrame) and df_cellclass is not None:
        raise ValueError(f"Argument 'df_cellclass' expected (None | pd.DataFrame) as input, received '{type(df_cellclass)}' instead!")

    # split the CNV DataFrame into its 2 sections --> multi-index and cells
    slice_data_list = ["CHR", "START", "END"]
    col_data = [c for c in df_cnv.columns if c not in slice_data_list]

    # df_cellclass | sorting the data by tags
    # ------------------------------------------------------------------------------------------------------------------
    # additional data in form of cell classes (col 1 of 2)
    if df_cellclass is not None:
        df_cellclass = df_cellclass.fillna("unknown")
        idx_cellclass = list(df_cellclass.index)
        union_cells = [c_tag for c_tag in idx_cellclass if c_tag in col_data]
        if not len(union_cells) > 0:
            raise ValueError("Provided DataFrame 'df_cellclass' must have a union with the CNV-matrix > 0!")

        # if union > 0 --> use union_cells as col_data list
        df_cellclass = df_cellclass.loc[union_cells, :]
        flag_sort = False
        if df_cellclass_classify_by is not None:
            if not str(df_cellclass_classify_by) in df_cellclass.columns:
                raise ValueError(f"Column '{df_cellclass_classify_by}' is not in DataFrame 'df_cellclass'!")
        else:
            df_cellclass_classify_by = df_cellclass.columns[0]
        if df_cellclass_filter_col:
            df_cellclass = df_cellclass[df_cellclass_classify_by]
        # sort if passed
        if sort:
            unique_list_col = list(df_cellclass[df_cellclass_classify_by].unique())
            list_sub_slices = [sort_df_row_by_similarity(df_cnv[df_cellclass[df_cellclass[df_cellclass_classify_by] == unique_tag].dropna(subset=df_cellclass_classify_by).index].T).T for unique_tag in unique_list_col]
            slice_data = pd.concat(list_sub_slices, axis=1)
            df_cellclass = df_cellclass.loc[list(slice_data.columns), :]
            flag_sort = True
        col_data = list(df_cellclass.index)
        print(f"> Matching, {"filtering, and sorting" if flag_sort else "and filtering"} of cell-classification [✓]")
        # true values of CNV method
        slice_data = df_cnv[col_data]
    else:
        if sort:
            # true values of CNV method
            slice_data = sort_df_row_by_similarity(df_cnv[col_data].T).T
            col_data = list(slice_data.columns)
            print(
            f"> Sorting data [✓]")
        else:
            # true values of CNV method
            slice_data = df_cnv[col_data]
    # ------------------------------------------------------------------------------------------------------------------

    # min-max normalization of the data
    # ---------------------------------
    max_val = abs(slice_data.max().max())
    min_val = abs(slice_data.min().min())
    # centering 0
    if max_val < min_val:
        max_val = min_val

    # normalized dataframe --> 0-1 min-max
    df_norm = slice_data.map(lambda x: round(x / max_val, 2))
    print("> normalization of data [✓]")


    # position of bins
    bin_df = df_cnv[slice_data_list]

    #########################
    # HOVER TEXT & Plotting #
    #########################

    # plotting element sizes  [px]
    # ----------------------------
    chr_bin_height = 60
    space_top_gene = 10
    gene_height = 30
    space_bottom_gene = 20
    sum_cna_height = 30 + 30*(len(df_cellclass[df_cellclass_classify_by].unique()) if df_cellclass is not None else 0)
    space_bottom_sum_cna = 10
    main_cna_height = len(col_data) if len(col_data) <= 1000 else 1000
    space_top_chr = 90
    chr_length = 600
    space_bottom_chr = 170
    table_height = 350


    vstack_height = (table_height
                     +main_cna_height
                     +sum_cna_height
                     +gene_height
                     +chr_bin_height
                     +chr_length
                     +space_top_chr
                     +space_bottom_chr
                     +space_top_gene
                     +space_bottom_gene
                     +space_bottom_sum_cna)

    list_genomic_pos_hm_tile = [f"{row["CHR"]} | {int(row["START"])} - {int(row["END"])}" for _, row in
                                bin_df.iterrows()]

    df_gene_info, _ = df_cna_idx_get_gene_info(bin_df, assembly_genome)

    list_genes_text = df_gene_info["genes_txt"].tolist()  # list with all genes per bin
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
    list_chr_info_update = [f"<b>{chr_tag}</b><br>   {arm}" for chr_tag, arm in zip(list_chr_info, list_genomic_arm_tag)]
    dict_chr_arm_encode = {"chr-arm | p": -0.3,"chr-arm | q": -0.1}
    list_chr_arm_val = list(map(dict_chr_arm_encode.get, list_genomic_arm_tag))

    print("> mapping of chromosomal bins [✓]")

    # create table underneath plot
    df_summary, dict_chr_bar = calc_pred_saturation(bin_df, assembly_genome)
    print("> generation of summary table [✓]")

    # create a multiplot
    # ------------------
    # std settings
    col_pos = 1
    col_width = [1]

    tuple_title = ("",
                   "",
                   "",
                   "",
                   "",
                   "",
                   "",
                   "",
                   "<b>Predictive saturation along the genome</b>",
                   "",
                   "")

    # change col layout if additional data
    if df_cellclass is not None:
        col_pos = 2
        col_width = [0.02*len(df_cellclass.columns), 0.9-0.02*len(df_cellclass.columns)]
        tuple_title = ("", "",
                       "", "",
                       "", "",
                       "", "",
                       "", "",
                       "", "",
                       "", "",
                       "", "",
                       "", "<b>Predictive saturation along the genome</b>",
                       "", "",
                       "", "")

    fig_vstack = make_subplots(rows=11,
                               cols=col_pos,
                               vertical_spacing=0,
                               subplot_titles=tuple_title,
                               horizontal_spacing=0.01,
                               row_heights=[chr_bin_height/vstack_height,
                                            space_top_gene/vstack_height,
                                            gene_height/vstack_height,
                                            space_bottom_gene/vstack_height,
                                            sum_cna_height/vstack_height,
                                            space_bottom_sum_cna/vstack_height,
                                            main_cna_height/vstack_height,
                                            space_top_chr/vstack_height,
                                            chr_length/vstack_height,
                                            space_bottom_chr/vstack_height,
                                            table_height/vstack_height],
                               column_widths=col_width,
                               specs=[[{"type": "xy"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "bar"}]*col_pos,
                                      [{"type": "xy"}]*col_pos,
                                      [{"type": "table"}]*col_pos]
                               )

    # top plot describing chromosome positions and genes located at the respective bins
    # colors --> alternating
    bin_size = len(df_norm)

    # bottom bin (genes for each genomic region)
    bin_color_list = [(x%2)*0.1 + 0.5 for x in range(1, bin_size+1, 1)]
    bin_hover_list = [f"<b>GenoSeg | {i+1}</b><br>   {txt[0]}<br>   {txt[1]}<br>   genes:<br>      {txt[2]}" for i, txt in
                      enumerate(zip(list_genomic_pos_hm_tile, list_genomic_arm_tag, list_genes_text))]
    print("> generation of hover text [✓]")

    # chromosome plot
    # ---------------
    top_fig = px.imshow([list_chr_val, list_chr_arm_val][::-1],
                        y=["<b>Chromosomes</b>", "<b>Chrom. arm</b>"][::-1],
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
                            y=["<b>Genes</b>"],
                            x=list_genomic_pos_hm_tile,
                            text_auto=False,
                            aspect="auto")
    middle_fig.update(data=[{"customdata": [bin_hover_list],
                          "hovertemplate":"%{customdata} <extra></extra>"}])

    fig_vstack.add_trace(middle_fig.data[0],
                         row=3,
                         col=col_pos)

    # main-sum CNA plot
    # -----------------
    mean_array_list = [df_norm.T.mean(axis=0).to_numpy()]
    sum_y = ["<b>summarized CNA | all cells</b>"]
    if df_cellclass is not None:
        unique_list = list(df_cellclass[df_cellclass_classify_by].unique())
        for unique_class in unique_list:
            filter_idx = list(df_cellclass.where(df_cellclass[df_cellclass_classify_by] == unique_class).dropna().index)
            mean_array_list.append(df_norm[filter_idx].T.mean(axis=0).to_numpy())
        sum_y.extend([f"<b>summarized CNA | {unique_class}</b>" for unique_class in unique_list])

    cna_fig = px.imshow(np.round(np.array(mean_array_list), decimals=2),
                    y=sum_y,
                    x=list_genomic_pos_hm_tile,
                    text_auto=False,
                    aspect="auto")
    cna_fig.update(data=[{"customdata": [list_text_t[0]]*len(sum_y),
                          "hovertemplate":"<b>Cell | %{y} </b><br>   val-CNA | %{z} <br>   %{x} <br>   %{customdata} <extra></extra>"}])

    fig_vstack.add_trace(cna_fig.data[0],
                         row=5,
                         col=col_pos)

    # main CNA plot
    # -------------
    cna_fig = px.imshow(df_norm.T.to_numpy(),
                    y=list(df_norm.columns),
                    x=list_genomic_pos_hm_tile,
                    text_auto=False,
                    aspect="auto")
    cna_fig.update(data=[{"customdata": list_text_t,
                          "hovertemplate":"<b>Cell | %{y} </b><br>   val-CNA | %{z} <br>   %{x} <br>   %{customdata} <extra></extra>"}])

    fig_vstack.add_trace(cna_fig.data[0],
                         row=7,
                         col=col_pos)

    # expansion plot for showing cell characteristics
    # -----------------------------------------------
    if df_cellclass is not None:
        # encode the classes
        dict_encode = encode_df_name_cols(df_cellclass)
        df_cellclass_sub = df_cellclass.copy()
        for col_tag, dict_encode_col in dict_encode.items():
            df_cellclass_sub[col_tag] = df_cellclass[col_tag].map(lambda x: dict_encode_col[x])

        # bring df_cellclass_classify_by to front
        df_cellclass_cols = list(df_cellclass.columns)
        df_cellclass_cols.insert(0, df_cellclass_cols.pop(df_cellclass_cols.index(df_cellclass_classify_by)))
        df_cellclass = df_cellclass[df_cellclass_cols]
        cellclass_info_arr = df_cellclass.to_numpy()

        # main support describe
        # ---------------------
        col_list_df_cellclass = [f"<b>{idx}</b>" for idx in list(df_cellclass.columns)]
        cell_fig = px.imshow(df_cellclass_sub[df_cellclass_cols].to_numpy(),
                            y=list(df_cellclass.index),
                            x=col_list_df_cellclass,
                            text_auto=False,
                            aspect="auto")

        cell_fig.update(data=[{"customdata": cellclass_info_arr,
                              "hovertemplate": "<b>Cell | %{y} </b><br>   %{customdata} <extra></extra>"}])

        fig_vstack.add_trace(cell_fig.data[0],
                             row=7,
                             col=1)
        fig_vstack.data[-1].update(coloraxis="coloraxis2")

        # summary support describe
        # ------------------------
        unique_list = list(df_cellclass[df_cellclass_classify_by].unique())
        list_val_summary = [[1.0]]
        list_val_summary.extend([[float(dict_encode[df_cellclass_classify_by][tag])] for tag in unique_list])
        sum_fig = px.imshow(np.array(list_val_summary),
                             y=sum_y,
                             x=["categories"],
                             text_auto=False,
                             aspect="auto")

        sum_fig.update(data=[{"hovertemplate": "<b>%{y}</b><extra></extra>"}])

        fig_vstack.add_trace(sum_fig.data[0],
                             row=5,
                             col=1)
        fig_vstack.data[-1].update(coloraxis="coloraxis2")

    # chromosomal barplot
    # -------------------
    chr_tag_barplot_bold = [f"<b>{chr_tag}</b>" for chr_tag in dict_chr_bar["chr"]]

    chr_multi_bar = [go.Bar(x=chr_tag_barplot_bold , y=dict_chr_bar["q_abs"],
                            customdata=dict_chr_bar["q_info"],
                            hovertemplate="<b>%{x} | q-arm</b><br>%{customdata}<extra></extra>",
                            marker_color='rgb(69, 73, 135)',
                            marker_line_color='rgba(255,255,255,1)',
                            marker_line_width=2,
                            marker=dict(cornerradius="100%")),
                     go.Bar(x=chr_tag_barplot_bold , y=dict_chr_bar["q"],
                            customdata=dict_chr_bar["q_info"],
                            hovertemplate="<b>%{x} | q-arm</b><br>%{customdata}<extra></extra>",
                            marker_color='rgb(242, 48, 129)',
                            marker_line_color='rgba(255,255,255,1)',
                            marker_line_width=1,
                            marker=dict(cornerradius="100%")),
                     go.Bar(x=chr_tag_barplot_bold , y=dict_chr_bar["p_abs"],
                            customdata=dict_chr_bar["p_info"],
                            hovertemplate="<b>%{x} | p-arm</b><br>%{customdata}<extra></extra>",
                            marker_color='rgb(69, 73, 135)',
                            marker_line_color='rgba(255,255,255,1)',
                            marker_line_width=2,
                            marker=dict(cornerradius="100%")),
                     go.Bar(x=chr_tag_barplot_bold , y=dict_chr_bar["p"],
                            customdata=dict_chr_bar["p_info"],
                            hovertemplate="<b>%{x} p-arm</b><br>%{customdata}<extra></extra>",
                            marker_color='rgb(53, 185, 242)',
                            marker_line_color='rgba(255,255,255,1)',
                            marker_line_width=1,
                            marker=dict(cornerradius="100%"))]

    # the plural of add_trace since a list of items! --> often a silly mistake
    fig_vstack.add_traces(chr_multi_bar,
                          rows=9,
                          cols=col_pos)

    # table
    # -----
    table_summary = go.Table(
        header=dict(
            values=list(df_summary.columns),
            line_color='rgb(34, 37, 87)',
            fill=dict(color="rgb(242, 48, 129)"),
            font=dict(size=15, color="white", weight="bold"),
            align="center"
        ),
        cells=dict(
            values=[df_summary[c].tolist() for c in df_summary.columns],
            line_color='rgb(34, 37, 87)',
            fill=dict(color="rgb(207, 70, 127)"),
            font=dict(size=12, color="white"),
            align="center")
    )

    fig_vstack.add_trace(table_summary,
                          row=11,
                          col=col_pos)

    ##########################
    # settings of fig_vstack #
    ##########################
    fig_vstack.update_xaxes(showticklabels=False, tickangle=-45, tickfont=dict(size=15))
    fig_vstack.update_layout(coloraxis2=dict(colorscale=[[0, "#009bde"],
                                                         [0.2, "#533fb5"],
                                                         [0.35, "#f23081"],
                                                         [0.5, "#cf3d36"],
                                                         [0.65, "#c98b3e"],
                                                         [0.8, "#81b349"],
                                                         [1.0, "#409c49"]],
                                             showscale=False),
                             coloraxis=dict(colorscale=[[0, "#009bde"],
                                                        [0.5, "#dedede"],
                                                        [1.0, "#f23081"]],
                                            colorbar=dict(len=(main_cna_height
                                                               +sum_cna_height
                                                               +space_top_gene
                                                               +space_bottom_gene
                                                               +space_bottom_sum_cna
                                                               -20),
                                                          lenmode="pixels",
                                                          title="<b>CNA-value</b>",
                                                          yanchor="top",
                                                          y=1,
                                                          dtick=0.25,
                                                          labelalias={1: "<b>gain</b>",
                                                                      0: "<b>neutral</b>",
                                                                      -1: "<b>loss</b>"})),
                             height=vstack_height,
                             showlegend=False,
                             hoverlabel=dict(font=dict(color='white')),
                             title_font_weight="bold",
                             title_font_size=38,
                             title_text=f"InferCNA plot {f'| {data_title}' if data_title is not None else ""}",
                             font=dict(color='rgb(222, 222, 222)'),
                             plot_bgcolor='rgb(34, 37, 87)',
                             paper_bgcolor='rgb(34, 37, 87)',
                             barmode="overlay")

    # lock cell-tag axis horizontally and vertically --> it adds a grid axis ID --> so any name is possible

    fig_vstack.update_xaxes(row=1, col=col_pos, matches='x')
    fig_vstack.update_xaxes(row=3, col=col_pos, matches='x')
    fig_vstack.update_xaxes(row=5, col=col_pos, matches='x')
    fig_vstack.update_xaxes(row=7, col=col_pos, matches='x')

    # overlay barplot
    fig_vstack.update_xaxes(row=9, col=col_pos, showgrid=False, showticklabels=True)
    fig_vstack.update_yaxes(row=9, col=col_pos, showgrid=False, showticklabels=False)

    if df_cellclass is not None:
        # main CNA
        # --------
        fig_vstack.update_yaxes(row=7, col=2, matches='y7', showticklabels=False)
        fig_vstack.update_yaxes(row=7, col=1, matches='y7', showticklabels=True)
        fig_vstack.update_xaxes(row=7, col=1, showticklabels=True)

        # summary CNA
        # -----------
        fig_vstack.update_yaxes(row=5, col=2, matches='y5', showticklabels=False)
        fig_vstack.update_yaxes(row=5, col=1, matches='y5', showticklabels=True)


    print("> generation of html document [✓]")

    # plot, show, and save figure
    # ---------------------------
    plotly.offline.plot(fig_vstack , filename=str(path_out / f"plot__{str(data_title).replace(" ", "_") if data_title is not None else "infercnv"}.html"))


# docker testing
if __name__ == "__main__":
    path_scevan_cnv = Path("/home/feilerwe/dashPlotly/data/plotlyCNA_capabilities/scevan_pred/out__GSE185269_10x_GBM__kerl_fixed_probes__hg_38__RCM__c5ngc5pg10bv0.5__scevan_")
    build_cnv_heatmap(df_cnv=pd.read_csv(path_scevan_cnv / "GSE185269_10x_GBM__kerl_fixed_probes__hg_38__RCM__c5ngc5pg10bv0.5__scevan__GBC__cur.csv").iloc[:, :1000],
                      df_cellclass=pd.read_csv(path_scevan_cnv / "GSE185269_10x_GBM__kerl_fixed_probes__hg_38__RCM__c5ngc5pg10bv0.5__scevan___results.csv", index_col="cells").iloc[:1000, :],
                      data_title="SCEVAN GBM filtered by fixed probes",
                      sort=True)
