# import
# ----------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import multiprocessing as mp
import os

from statsmodels.graphics.tsaplots import plot_predict

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
# -----------------
general_data_path = Path(__file__).parent.parent / "data" / "genome"


def list_transpose(l: list) -> list:
    return list(map(list, zip(*l)))


def convert_ucsc_cytoband(path: (str | Path)) -> pd.DataFrame:
    df_cytoband = pd.read_csv(Path(path), sep="\t",
                              names=["CHR", "START", "END", "tag", "tag_alter"]).dropna(how="any", axis=0)
    set_chr = list(df_cytoband["CHR"].unique())
    list_concat = []
    for chr_tag in set_chr:
        slice_df = df_cytoband.where(df_cytoband["CHR"] == chr_tag).dropna().sort_values(by="START", ascending=True)
        slice_df["tag_arm"] = [list(t)[0] for t in list(slice_df["tag"])]
        slice_df.drop_duplicates(subset="tag_arm", keep="first", inplace=True)
        list_concat.append(slice_df)
    return (pd.concat(list_concat).astype({"START":int, "END": int}).set_index("CHR").
            drop(["tag", "tag_alter", "END"], axis=1))


def map_arm_by_chr_pos(bin_df: pd.DataFrame, df_arms: pd.DataFrame):
    list_genomic_arm_tag = [f"chr-arm | {"p" if df_arms.where(df_arms["tag_arm"] == "q").dropna().
                                                loc[row["CHR"], "START"] > row["START"] else "q"}" for _, row in
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

    # import data
    df_arms = pd.read_csv(general_data_path / f"{assembly_genome}__loc_qp_arm.csv")

    # check input
    for col_tags in slice_data_list:
        if not col_tags in available_cols :
            raise ValueError(f"{slice_data_list} must be contained in DataFrame columns! Only {available_cols} were found.")
    df_cna_idx["DIFF"] = df_cna_idx["END"] - df_cna_idx["START"]
    df_cna_idx_unique_chr_list = list(df_cna_idx["CHR"].unique())
    df_chr_total_len = pd.read_csv(Path(general_data_path / f"{assembly_genome}__chr_bin_lengths__ucsc.csv"),
                                   index_col="CHR")
    dict_df = {"":["chrom<br>pred<br>saturation", "p-arm<br>pred<br>saturation", "q-arm<br>pred<br>saturation"]}
    list_len_p_abs = []
    list_len_q_abs = []
    list_len_p = []
    list_len_q = []
    info_p_arm_list = []
    info_q_arm_list = []
    info_chr_list = []
    for idx, row in df_chr_total_len.iterrows():
        if not idx in df_cna_idx_unique_chr_list:
            continue
        info_chr_list.append(str(idx).replace("chr", "Chromosome "))
        breakpoint_q = df_arms.where(df_arms["CHR"] == idx).dropna().set_index("tag_arm").loc["q", "START"]
        list_df_filter = [(df_cna_idx.where(df_cna_idx["CHR"] == idx).dropna(), row["bin_size"]),
                          (df_cna_idx.where((df_cna_idx["CHR"] == idx) & (df_cna_idx["END"] <= breakpoint_q)).dropna(), breakpoint_q),
                          (df_cna_idx.where((df_cna_idx["CHR"] == idx) & (df_cna_idx["START"] >= breakpoint_q)).dropna(), row["bin_size"]-breakpoint_q)]
        list_percentages_per_section = []
        for section_df, chr_bin_size in list_df_filter:
            chr_slice_sum_diff = sum(list(section_df["DIFF"]))
            chr_diff_percent = round((chr_slice_sum_diff/chr_bin_size)*100, 2)
            list_percentages_per_section.append(f"{chr_diff_percent} %")
        p_abs = list_df_filter[1][1]
        q_abs = list_df_filter[2][1]
        p_pred = sum(list(list_df_filter[1][0]["DIFF"]))
        q_pred = sum(list(list_df_filter[2][0]["DIFF"]))
        list_len_p_abs.append(p_abs)
        list_len_q_abs.append(-q_abs)
        list_len_p.append(p_pred)
        list_len_q.append(-q_pred)
        info_p_arm_list.append(f"total_length | {p_abs} bp<br> pred | {p_pred} bp<br>pred-% | {list_percentages_per_section[1]}")
        info_q_arm_list.append(f"total_length | {q_abs} bp<br> pred | {q_pred} <br>pred-% | {list_percentages_per_section[2]}")
        dict_df[f"<br>{str(idx)}"] = list_percentages_per_section

    total_diff_percent = round((sum(list(df_cna_idx["DIFF"]))/sum(list(df_chr_total_len["bin_size"])))*100, 2)
    dict_df["total<br>Genome"] = [f"{total_diff_percent} %", " - ", " - "]
    return pd.DataFrame.from_dict(dict_df), {"chr": info_chr_list,
                                             "q_info": info_q_arm_list,
                                             "q":list_len_q,
                                             "q_abs":list_len_q_abs,
                                             "p_info": info_p_arm_list,
                                             "p":list_len_p,
                                             "p_abs":list_len_p_abs}


# Debugging
if __name__ == "__main__":
    pass