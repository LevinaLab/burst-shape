def make_target_label(
    dataset,
    df_cultures,
    df_bursts=None,
    special_target=False,
    target_column_name="target_label",
):
    match dataset:
        case "inhibblock":
            target_label = "drug_label"
        case "kapucu":
            target_label = "culture_type"
        case "wagenaar":
            target_label = "batch"
        case "mossink":
            if special_target is True:
                target_label = "group-subject"
                df_cultures.reset_index(inplace=True)
                df_cultures["group-subject"] = (
                    df_cultures["group"] + " " + df_cultures["subject_id"].astype(str)
                )
                # df_cultures.set_index(["group-subject", "well_idx"], inplace=True)
                df_cultures.set_index(["group", "subject_id", "well_idx"], inplace=True)

                if df_bursts is not None:
                    df_bursts.reset_index(inplace=True)
                    df_bursts["group-subject"] = (
                        df_bursts["group"] + " " + df_bursts["subject_id"].astype(str)
                    )
                    # df_bursts.set_index(
                    #     ["group-subject", "well_idx", "i_burst"], inplace=True
                    # )
                    df_bursts.set_index(
                        ["group", "subject_id", "well_idx", "i_burst"], inplace=True
                    )
            else:
                target_label = "group"
        case _:
            raise NotImplementedError(
                f"Target label is not implemented for dataset {dataset}."
                "Go to this function and define a target label."
            )
    df_cultures[target_column_name] = df_cultures.reset_index()[target_label].values
    if df_bursts is not None:
        df_bursts[target_column_name] = df_bursts.reset_index()[target_label].values
    if df_bursts is not None:
        return df_cultures, df_bursts, target_label
    else:
        return df_cultures, target_label
