import numpy as np
import pandas as pd
import scipy.stats as stats
from clearml import Logger
from matplotlib import pyplot as plt

from utilz import *

GLOBAL_LOG_DEFAULT = False
EPSILON = 0.000001

def calculate_aggregate_df(
    aggregate_df_data_df: pd.DataFrame,
    selected_error_bound: float,
    selection_criteria: str = "original",
    log_time=GLOBAL_LOG_DEFAULT,
):
    before = DateUtils.now()

    
    selected_aggregate_df_data_df, aggregate_df = None, None
    try:
        if selection_criteria == "original":
            selected_aggregate_df_data_df = aggregate_df_data_df[
                aggregate_df_data_df["mean_mse"] <= selected_error_bound
            ]
            aggregate_df = selected_aggregate_df_data_df.copy()
        elif selection_criteria.startswith("q_"):
            quantile_value = float(selection_criteria.split("_")[1]) / 100.0
            if not 0 < quantile_value < 1:
                raise ValueError("quantile_value must be between 0 and 1.")
            z_score = stats.norm.ppf(quantile_value)
            aggregate_df = aggregate_df_data_df.copy()
            aggregate_df[f"mean_mse_{selection_criteria}"] = (
                aggregate_df["mean_mse"] + z_score * aggregate_df["std"]
            )
            aggregate_df[f"threshold_{selection_criteria}"] = selected_error_bound
            
            selected_aggregate_df_data_df = aggregate_df[
                aggregate_df[f"mean_mse_{selection_criteria}"] <= selected_error_bound
            ].copy()
        else:
            raise ValueError(f"Selection criteria '{selection_criteria}' unsupported!")
    except Exception as e:
        raise e

    return selected_aggregate_df_data_df, aggregate_df



def calculate_energy_bounds(
    df_total_filtered,
    energy_key: str,
    mse_key: str,
    step_size: int,
    log_time=GLOBAL_LOG_DEFAULT,
):
    before = DateUtils.now()
    min_noisy_energy = df_total_filtered[energy_key].min()
    max_noisy_energy = df_total_filtered[energy_key].max()

    list_of_aggregate_df_data = []

    def to_int_fallback(value, fallback_value):
        try:
            return int(value)
        except (ValueError, TypeError):
            print(f"Falling back to {fallback_value}...")
            return fallback_value

    try:
        range_start = to_int_fallback(
            min_noisy_energy * step_size, fallback_value=-1 * step_size
        )
        range_end = to_int_fallback(
            max_noisy_energy * step_size, fallback_value=range_start + (2 * step_size)
        )
        range_of_energies = range(range_start, range_end, 1)
    except Exception as e:
        print(
            f"Exception has occurred while calculating range_of_energies. Exception message: {e}"
        )
        raise e
        

    if len(range_of_energies) == 1 or (
        range_of_energies.start == range_of_energies.stop
    ):
        import math

        range_of_energies = range(
            int(math.floor(min_noisy_energy) * step_size),
            int(math.ceil(max_noisy_energy) * step_size) + 2,
            1,
        )

    def len_to_str(length):
        if length < 25:
            return "< 25"
        if length < 100:
            return "< 100"
        elif length < 500:
            return "< 500"
        elif length < 1000:
            return "< 1000"
        elif length < 10000:
            return "< 10000"
        elif length < 25000:
            return "< 25000"
        elif length >= 25000:
            return ">= 25000"

    for i in range_of_energies:
        subset_df = df_total_filtered[
            (df_total_filtered[energy_key] >= i / step_size)
            & (df_total_filtered[energy_key] < ((i + 1) / step_size))
        ]
        aggregate_df_obj = {
            "energy_range_idx": i,
            "energy_range": i / step_size,
            "mean_mse": subset_df[mse_key].mean(),
            "std": subset_df[mse_key].std(),
            "count": len_to_str(len(subset_df[mse_key])),
            "actual_count": (len(subset_df[mse_key])),
        }
        list_of_aggregate_df_data.append(aggregate_df_obj)

    global aggregate_df_data_df_with_na, aggregate_df_data_df, selected_aggregate_df_data_df
    aggregate_df_data_df_with_na = pd.DataFrame(list_of_aggregate_df_data)
    aggregate_df_data_df = aggregate_df_data_df_with_na.dropna()

    return aggregate_df_data_df, range_of_energies


def calculate_energy_bounds_optimized(
    df_total_filtered,
    energy_key: str,
    mse_key: str,
    step_size: int,
    log_time=GLOBAL_LOG_DEFAULT,
):
    """
    TODO: This method is not used in the current implementation. Need to see if it is robust against weird errors
    """

    before = DateUtils.now()

    
    min_noisy_energy = df_total_filtered[energy_key].min()
    max_noisy_energy = df_total_filtered[energy_key].max()

    
    energy_bins = (
        np.arange(
            int(min_noisy_energy * step_size), int(max_noisy_energy * step_size) + 2, 1
        )
        / step_size
    )

    
    energy_categories = pd.cut(df_total_filtered[energy_key], bins=energy_bins)

    
    grouped = df_total_filtered.groupby(energy_categories)
    aggregate_df_data = grouped.agg({mse_key: ["mean", "std", "count"]}).reset_index()

    
    aggregate_df_data.columns = ["energy_range", "mean_mse", "std", "actual_count"]

    
    aggregate_df_data["energy_range_idx"] = range(len(aggregate_df_data))

    
    def len_to_str(length):
        if length < 25:
            return "< 25"
        if length < 100:
            return "< 100"
        elif length < 500:
            return "< 500"
        elif length < 1000:
            return "< 1000"
        elif length < 10000:
            return "< 10000"
        elif length < 25000:
            return "< 25000"
        else:
            return ">= 25000"

    aggregate_df_data["count"] = aggregate_df_data["actual_count"].apply(len_to_str)

    
    aggregate_df_data = aggregate_df_data.sort_values("energy_range").reset_index(
        drop=True
    )

    aggregate_df_data = aggregate_df_data.dropna()
    return aggregate_df_data, energy_bins


def calculate_selected_and_filtered_dfs(
    selected_aggregate_df_data_df: pd.DataFrame,
    range_of_energies,
    step_size,
    log_time=GLOBAL_LOG_DEFAULT,
):
    before = DateUtils.now()

    
    def calculate_selected_and_filtered_dfs_old():
        energy_step_size = (range_of_energies[1] - range_of_energies[0]) / step_size

        def check_if_in_interval(
            energy,
            energy_ranges=selected_aggregate_df_data_df["energy_range"],
            energy_step_size=energy_step_size,
        ):
            for energy_range in energy_ranges:
                range_start = energy_range
                range_end = energy_range + energy_step_size
                if energy >= range_start and energy < range_end:
                    return True
            return False

        selected_df_total_orig_model = df_total_orig_model_only[
            df_total_orig_model_only.apply(
                lambda row: check_if_in_interval(row["noisy_energies"]), axis=1
            )
        ]
        filtered_df_total_orig_model = df_total_orig_model_only[
            df_total_orig_model_only.apply(
                lambda row: not check_if_in_interval(row["noisy_energies"]), axis=1
            )
        ]

        
        return selected_df_total_orig_model, filtered_df_total_orig_model

    

    def calculate_selected_and_filtered_dfs_vectorized():
        if len(selected_aggregate_df_data_df) == 0:
            
            selected_df_total_orig_model_new = df_total_orig_model_only[
                [False] * len(df_total_orig_model_only.index)
            ]
            filtered_df_total_orig_model_new = df_total_orig_model_only[
                [True] * len(df_total_orig_model_only.index)
            ]
            return (
                selected_df_total_orig_model_new,
                filtered_df_total_orig_model_new,
            )
        
        
        energy_ranges = selected_aggregate_df_data_df["energy_range"].values
        energy_step_size = (range_of_energies[1] - range_of_energies[0]) / step_size

        
        range_bounds = np.column_stack(
            (energy_ranges, energy_ranges + energy_step_size)
        )

        
        noisy_energies = df_total_orig_model_only["noisy_energies"].values[
            :, np.newaxis
        ]
        in_interval = np.any(
            (noisy_energies >= range_bounds[:, 0])
            & (noisy_energies < range_bounds[:, 1]),
            axis=1,
        )

        
        selected_df_total_orig_model_new = df_total_orig_model_only[in_interval]
        filtered_df_total_orig_model_new = df_total_orig_model_only[~in_interval]
        return selected_df_total_orig_model_new, filtered_df_total_orig_model_new

    (
        selected_df_total_orig_model,
        filtered_df_total_orig_model,
    ) = calculate_selected_and_filtered_dfs_vectorized()

    return selected_df_total_orig_model, filtered_df_total_orig_model


def generate_select_result_obj(
    selected_df_total_orig_model: pd.DataFrame,
    filtered_df_total_orig_model: pd.DataFrame,
    selected_error_bound: float,
    log_time=GLOBAL_LOG_DEFAULT,
):
    before = DateUtils.now()

    val_mse_of_selected_model = selected_df_total_orig_model["noisy_mse"].mean()
    val_std_of_selected_model = selected_df_total_orig_model["noisy_mse"].std()
    val_mse_of_filtered_model = filtered_df_total_orig_model["noisy_mse"].mean()
    val_std_of_filtered_model = filtered_df_total_orig_model["noisy_mse"].std()
    val_mse_of_orig_model = df_total_orig_model_only["noisy_mse"].mean()
    val_std_of_orig_model = df_total_orig_model_only["noisy_mse"].std()

    orig_model_count_of_satisfying_error_bound = len(
        df_total_orig_model_only[
            df_total_orig_model_only["noisy_mse"] <= selected_error_bound
        ]
    )
    selected_model_count_of_satisfying_error_bound = len(
        selected_df_total_orig_model[
            selected_df_total_orig_model["noisy_mse"] <= selected_error_bound
        ]
    )

    orig_model_proportion_of_satisfying_error_bound = (
        orig_model_count_of_satisfying_error_bound
        / (len(df_total_orig_model_only) + 0.0001)
    )
    selected_model_proportion_of_satisfying_error_bound = (
        selected_model_count_of_satisfying_error_bound
        / (len(selected_df_total_orig_model) + 0.0001)
    )

    true_positive_df = selected_df_total_orig_model[
        selected_df_total_orig_model["noisy_mse"] <= selected_error_bound
    ]
    false_positive_df = selected_df_total_orig_model[
        selected_df_total_orig_model["noisy_mse"] > selected_error_bound
    ]
    false_negative_df = filtered_df_total_orig_model[
        filtered_df_total_orig_model["noisy_mse"] <= selected_error_bound
    ]
    true_negative_df = filtered_df_total_orig_model[
        filtered_df_total_orig_model["noisy_mse"] > selected_error_bound
    ]

    sum_false_positive_error = (
        false_positive_df["noisy_mse"] - selected_error_bound
    ).sum()
    sum_true_negative_error = (
        true_negative_df["noisy_mse"] - selected_error_bound
    ).sum()

    mean_false_positive_error = sum_false_positive_error / (
        len(selected_df_total_orig_model) + 1
    )
    mean_false_negative_error = false_negative_df["noisy_mse"].sum() / (
        len(filtered_df_total_orig_model) + 1
    )
    mean_true_negative_error = sum_true_negative_error / (
        len(filtered_df_total_orig_model) + 1
    )

    true_positive_count = len(true_positive_df) + 1
    false_positive_count = len(false_positive_df) + 1
    false_negative_count = len(false_negative_df) + 1
    true_negative_count = len(true_negative_df) + 1

    f1_score = true_positive_count / (
        true_positive_count + 0.5 * (false_positive_count + false_negative_count)
    )

    
    select_result_obj = {
        "selected_error_bound": selected_error_bound,
        
        "val_mse_selected_model": val_mse_of_selected_model,
        "val_mse_orig_model": val_mse_of_orig_model,
        "delta_val_mse_model": val_mse_of_orig_model - val_mse_of_selected_model,
        "val_mse_of_filtered_model": val_mse_of_filtered_model,
        "val_std_of_selected_model": val_std_of_selected_model,
        "val_std_of_filtered_model": val_std_of_filtered_model,
        "val_std_of_orig_model": val_std_of_orig_model,
        "mean_true_negative_error": mean_true_negative_error,
        "mean_false_positive_error": mean_false_positive_error,
        "mean_false_negative_error": mean_false_negative_error,
        "sum_false_positive_error": sum_false_positive_error,
        "sum_true_negative_error": sum_true_negative_error,
        "true_positive_count": true_positive_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "true_negative_count": true_negative_count,
        "f1_score": f1_score,
        "selected_value_count": len(selected_df_total_orig_model),
        "filtered_value_count": len(filtered_df_total_orig_model),
        "orig_model_count_of_satisfying_error_bound": orig_model_count_of_satisfying_error_bound,
        "selected_model_count_of_satisfying_error_bound": selected_model_count_of_satisfying_error_bound,
        "orig_model_proportion_of_satisfying_error_bound": orig_model_proportion_of_satisfying_error_bound,
        "selected_model_proportion_of_satisfying_error_bound": selected_model_proportion_of_satisfying_error_bound,
        "delta_model_proportion_of_satisfying_error_bound": selected_model_proportion_of_satisfying_error_bound
        - orig_model_proportion_of_satisfying_error_bound,
    }

    return select_result_obj






def test_inference_strategy(
    noisy_std,
    step_size,
    error_bounds,
    error_bounds_percentages,
    result_object_to_use,
    result_object_train,
    parent_path_pics,
    select_only_one_std=False,
    use_agg=False,
    use_energy_centering=False,
    id=0,
    result_object_name="test",
    use_other_project=False,  
    max_num_of_noisy_samples: int = None,
    train_dataset_scaler=None,
    selection_criteria="original",
    output_pdf=False,
    is_test_mode=False,
):
    if max_num_of_noisy_samples is not None:
        # Keep this important parameter info
        print(f"Limited max_num_of_noisy_samples to {max_num_of_noisy_samples}")
    
    selected_metrics_file_name = f"selected_obj_std={noisy_std}"
    selected_metrics_file_name += f"_steps={step_size}"
    selected_metrics_file_name += f"_n_per={len(error_bounds_percentages)}"
    selected_metrics_file_name += f"_one_std={select_only_one_std}"
    selected_metrics_file_name += f"_agg={use_agg}"
    selected_metrics_file_name += f"_centering={use_energy_centering}"
    selected_metrics_file_name += f"_obj={result_object_name}"
    selected_metrics_file_name += f"_id={id}"
    selected_metrics_file_name += f"_samples={max_num_of_noisy_samples}"
    
    
    if is_test_mode:
        selected_metrics_file_name += "_test_mode"
        print(f"Running test_inference_strategy in test mode with reduced processing")
        
        
        if len(error_bounds) > 10:
            
            step = len(error_bounds) // 10
            error_bounds = error_bounds[::step]
            error_bounds_percentages = error_bounds_percentages[::step]
            print(f"Reduced error bounds from {len(error_bounds_percentages)} to {len(error_bounds)}")
            
        
        if max_num_of_noisy_samples and max_num_of_noisy_samples > 16:
            max_num_of_noisy_samples = 16
            print(f"Limited max_num_of_noisy_samples to {max_num_of_noisy_samples}")
    
    selected_pdf_file_name = selected_metrics_file_name + ".pdf"
    selected_csv_file_name = selected_metrics_file_name + ".csv"
    csv_path = f"{os.path.join(parent_path_pics, selected_csv_file_name)}"

    
    
        
        

    before_start = DateUtils.now()

    parent_path_pics_path = Path(parent_path_pics)
    parent_path_pics_name = parent_path_pics_path.name

    list_of_selected_model_metrics_objects = []

    error_bounds_count = len(error_bounds)
    for idx, selected_error_bound in enumerate(error_bounds):
        
        

        
        
        

        before_step_2 = DateUtils.now()

        selected_error_bound_percentage = error_bounds_percentages[idx]

        noisy_stds = np.unique(result_object_train["noisy_std"])
        selected_noisy_std_start_idx = 0

        if select_only_one_std:
            
            for noisy_std_idx, noisy_std_value in enumerate(
                result_object_to_use["noisy_std"][0]
            ):
                if noisy_std_value == noisy_std:
                    selected_noisy_std_start_idx = noisy_std_idx
                    break

        selected_noisy_std_end_idx = 0
        for noisy_std_idx, noisy_std_value in enumerate(
            result_object_to_use["noisy_std"][0]
        ):
            if noisy_std_value > noisy_std:
                selected_noisy_std_end_idx = noisy_std_idx - 1
                break

        if noisy_std == max(noisy_stds):
            selected_noisy_std_end_idx = len(result_object_to_use["noisy_std"])

        if max_num_of_noisy_samples is not None:
            selected_noisy_std_end_idx = min(
                [
                    (selected_noisy_std_start_idx + max_num_of_noisy_samples),
                    selected_noisy_std_end_idx,
                ]
            )

        global filtered_noisy_energies, recalculated_noisy_mse_y_hat, noisy_stds_for_df, noisy_energy_array
        if use_agg:
            noisy_energy_array = result_object_train["noisy_energies_y_hat"][
                :, selected_noisy_std_start_idx:selected_noisy_std_end_idx
            ]
            if use_energy_centering:
                noisy_energy_array = noisy_energy_array - result_object_train[
                    "energy_hats_init_orig_model"
                ].reshape(-1, 1)
            filtered_noisy_energies = noisy_energy_array.mean(axis=1).reshape(-1, 1)
            recalculated_noisy_mse_y_hat = (result_object_train["mse_init_orig_model"],)
            noisy_stds_for_df = np.ones_like(recalculated_noisy_mse_y_hat)
        else:
            noisy_energy_array = result_object_train["noisy_energies_y_hat"][
                :, selected_noisy_std_start_idx:selected_noisy_std_end_idx
            ]
            if use_energy_centering:
                noisy_energy_array = noisy_energy_array - result_object_train[
                    "energy_hats_init_orig_model"
                ].reshape(-1, 1)
            filtered_noisy_energies = noisy_energy_array
            recalculated_noisy_mse_y_hat = result_object_train["noisy_mse_y_hat"][
                :, selected_noisy_std_start_idx:selected_noisy_std_end_idx
            ] + result_object_train["mse_init_orig_model"].reshape(-1, 1)
            noisy_stds_for_df = result_object_train["noisy_std_y_hat"][
                :, selected_noisy_std_start_idx:selected_noisy_std_end_idx
            ]

        global df_total_filtered
        ENERGY_KEY = "noisy_energies_y_hat"
        MSE_KEY = "noisy_mse_y_hat"
        STD_KEY = "noisy_std_y_hat"
        df_total_filtered = pd.DataFrame(
            {
                ENERGY_KEY: np.concatenate(filtered_noisy_energies),
                MSE_KEY: np.concatenate(recalculated_noisy_mse_y_hat),
                STD_KEY: np.concatenate(noisy_stds_for_df),
            }
        )

        before_step_3 = DateUtils.now()

        
        
        aggregate_df_data_df, range_of_energies = calculate_energy_bounds(
            df_total_filtered=df_total_filtered,
            energy_key=ENERGY_KEY,
            mse_key=MSE_KEY,
            step_size=step_size,
        )

        
        
        
        
        
        

        def check_if_dataframes_are_equal():
            original = aggregate_df_data_df[
                ["mean_mse", "actual_count", "std", "count"]
            ]
            headed = aggregate_df_data_df_o.head(len(aggregate_df_data_df_o))[
                ["mean_mse", "actual_count", "std", "count"]
            ]
            dataframes_equal = original.round(3).equals(headed.round(3))
            if not dataframes_equal:
                raise ValueError("Dataframes are not equal")

        before_step_4 = DateUtils.now()

        
        

        (
            selected_aggregate_df_data_df,
            updated_aggregate_df_data_df,
        ) = calculate_aggregate_df(
            aggregate_df_data_df=aggregate_df_data_df,
            selected_error_bound=selected_error_bound,
            selection_criteria=selection_criteria,
        )

        
        selected_aggregate_df_data_df["sum_error"] = (
            selected_aggregate_df_data_df["mean_mse"]
            * selected_aggregate_df_data_df["actual_count"]
        )
        selected_mse = (
            selected_aggregate_df_data_df["sum_error"].sum()
            / selected_aggregate_df_data_df["actual_count"].sum()
        )
        train_coverage = (
            selected_aggregate_df_data_df["actual_count"].sum()
            / aggregate_df_data_df["actual_count"].sum()
        )

        before_step_5 = DateUtils.now()

        
        

        if use_agg:
            noisy_energy_array = result_object_to_use[ENERGY_KEY][
                :, selected_noisy_std_start_idx:selected_noisy_std_end_idx
            ]
            if use_energy_centering:
                noisy_energy_array = noisy_energy_array - result_object_to_use[
                    "energy_hats_init_orig_model"
                ].reshape(-1, 1)
            noisy_energies_to_test = noisy_energy_array.mean(axis=1)
        else:
            noisy_energy_array = np.concatenate(
                result_object_to_use["energy_hats_init_orig_model"].reshape(-1, 1)
            )
            
            
            
            noisy_energies_to_test = noisy_energy_array

        df_total_orig_model_only_object = {
            "noisy_energies": noisy_energies_to_test,
            "transformed_mse": np.concatenate(
                result_object_to_use["inverse_mse"].reshape(-1, 1)
            ),
            "noisy_mse": np.concatenate(
                result_object_to_use["mse_init_orig_model"].reshape(-1, 1)
            ),
        }

        global df_total_orig_model_only
        df_total_orig_model_only = pd.DataFrame(df_total_orig_model_only_object)

        
        

        before_selected_and_filtered = DateUtils.now()
        (
            selected_df_total_orig_model,
            filtered_df_total_orig_model,
        ) = calculate_selected_and_filtered_dfs(
            selected_aggregate_df_data_df=selected_aggregate_df_data_df,
            range_of_energies=range_of_energies,
            step_size=step_size,
        )

        after_selected_and_filtered = DateUtils.now()

        
        

        
        common_select_result_obj = generate_select_result_obj(
            selected_df_total_orig_model=selected_df_total_orig_model,
            filtered_df_total_orig_model=filtered_df_total_orig_model,
            selected_error_bound=selected_error_bound,
        )
        select_result_obj = {
            **common_select_result_obj,
            "selected_error_bound_percentage": selected_error_bound_percentage,
            "selection_id": id,
            "noisy_std": noisy_std,
            "step_size": step_size,
            "n_percentages": len(error_bounds_percentages),
            "select_only_one_std": select_only_one_std,
            "use_agg": use_agg,
            "use_energy_centering": use_energy_centering,
            "result_object_name": result_object_name,
            "max_num_of_noisy_samples": max_num_of_noisy_samples,
            "train_coverage": train_coverage,
            "selection_criteria": selection_criteria,
        }
        after_agg = DateUtils.now()
        list_of_selected_model_metrics_objects.append(select_result_obj)

    global selected_metrics_object_df
    selected_metrics_object_df = pd.DataFrame(list_of_selected_model_metrics_objects)

    if output_pdf:
        try:
            
            
            import seaborn as sns

            plt.figure(figsize=(12, 8), dpi=200)

            if selection_criteria != "original":
                p = sns.barplot(
                    x="energy_range",
                    y=f"mean_mse_{selection_criteria}",
                    hue="count",
                    data=updated_aggregate_df_data_df,
                    ci=None,
                    dodge=False,
                    palette="coolwarm",
                )
            else:
                p = sns.barplot(
                    x="energy_range",
                    y=f"mean_mse",
                    hue="count",
                    data=updated_aggregate_df_data_df,
                    ci=None,
                    dodge=False,  
                    palette="coolwarm",
                )

            try:
                for container in p.containers:
                    p.bar_label(
                        container,
                        labels=updated_aggregate_df_data_df["actual_count"],
                        label_type="center",
                        padding=-20,
                    )
            except:
                ...
                

            p.set_xlabel("Energy value [x; x+0.1)")
            p.set_ylabel("MSE of the sample")

            p.text(
                0.5,
                0.1,
                selected_metrics_file_name,
                size=8,
                weight="light",
                ha="center",
                va="center",
                transform=p.transAxes,
            )

            plt.legend(title="Sample count")
            plt.title(
                f"Noisy Energy correlation barplot with max noise std={noisy_std}"
            )

            
            plt.savefig(f"{os.path.join(parent_path_pics, selected_pdf_file_name)}")
        except ValueError as e:
            print(f"Error while saving selected_pdf file. e={e}")

    try:
        csv_path = f"{os.path.join(parent_path_pics, selected_csv_file_name)}"
        selected_metrics_object_df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error while saving selected_csv file. e={e}")

    return selected_metrics_object_df

def perform_selective_inference_experiments(
        result_object: dict,
        result_object_train: dict,
        result_object_test: dict,
        parent_path_pics: str,
        train_dataset_scaler,
        is_different_project=False,
        parallel_pool_size=None,
        is_test_mode=False,
        noisy_std_custom=None,
):
    """Simplified noise-based selective inference using direct energy sorting.

    New behaviour (for TEM noise strategy):
    - 不再使用噪声采样的 noisy_* 数组，也不做能量分箱。
    - 直接按能量从低到高排序样本，对一组目标 coverage 取前 k% 样本，
            在这些样本上计算 MSE，得到 coverage→MSE 曲线。

    返回的 DataFrame 仍然包含 train_coverage、val_mse_selected_model、val_mse_orig_model
    等列，以兼容后续写 CSV 和打印 summary 的逻辑。
    """

    try:
        # 使用验证集上的能量来确定能量阈值和 empirical coverage
        val_energy = np.asarray(result_object["energy_hats_init_orig_model"])
        val_mse_sel = np.asarray(result_object["mse_init_orig_model"])
        val_mse_orig = np.asarray(result_object["mse_orig"])

        # 测试集上的能量和误差（用于最终评估）
        test_energy = np.asarray(result_object_test["energy_hats_init_orig_model"])
        test_mse_sel = np.asarray(result_object_test["mse_init_orig_model"])
        test_mse_orig = np.asarray(result_object_test["mse_orig"])

        # 在验证集上按能量排序一次，后面复用排序结果
        val_order = np.argsort(val_energy)
        sorted_val_energy = val_energy[val_order]

        target_coverages = [0.1, 0.3, 0.5, 0.7, 0.9]

        def build_metrics_for_split(split_name, split_energy, split_mse_sel, split_mse_orig):
            rows = []
            n_val = len(sorted_val_energy)

            if n_val == 0:
                return pd.DataFrame(rows)

            for target_cov in target_coverages:
                k = max(1, int(round(target_cov * n_val)))
                k = min(k, n_val)

                # 在验证集上确定能量阈值（小于等于该能量的样本被选中）
                energy_threshold = sorted_val_energy[k - 1]

                val_mask = val_energy <= energy_threshold
                val_coverage = float(val_mask.sum()) / float(n_val)

                split_mask = split_energy <= energy_threshold
                if split_mask.sum() == 0:
                    # 没有样本被选中时，跳过该点
                    continue

                mse_selected = float(split_mse_sel[split_mask].mean())
                mse_orig_all = float(split_mse_orig.mean())

                row = {
                    "target_coverage": target_cov,
                    # 为兼容后续打印逻辑，这里字段名仍使用 train_coverage，
                    # 但实际含义已经是 "验证集上的 empirical coverage"。
                    "train_coverage": val_coverage,
                    "val_mse_selected_model": mse_selected,
                    "val_mse_orig_model": mse_orig_all,
                    "split": split_name,
                }
                rows.append(row)

            return pd.DataFrame(rows)

        # 验证集上也算一遍，方便查看；但阈值都是基于验证集本身
        val_df = build_metrics_for_split("val", val_energy, val_mse_sel, val_mse_orig)
        # 最终我们主要关心测试集上的表现
        test_df = build_metrics_for_split("test", test_energy, test_mse_sel, test_mse_orig)

        print(
            f"[INFO] Simple energy-sorting selective inference produced "
            f"{len(val_df)} val rows and {len(test_df)} test rows for coverages {target_coverages}"
        )

        return val_df, test_df, result_object

    except Exception as e:
        print(f"[ERROR] perform_selective_inference_experiments failed: {e}")
        return None, None, None


def test_inference_strategy_with_optim(
    inference_steps,
    inference_optim_lr,
    error_bounds,
    error_bounds_percentages,
    result_object_to_use,
    result_object_train,
    parent_path_pics,
    step_size=50,
    id=0,
    result_object_name="test",
    output_pdf=False,
    use_other_project=False,
    is_test_mode=False,
):
    selected_metrics_file_name = f"ebm_optim__steps={inference_steps}"
    selected_metrics_file_name += f"optim_lr_e={abs(int(np.log10(inference_optim_lr)))}"
    selected_metrics_file_name += f"_n_per={len(error_bounds_percentages)}"
    selected_metrics_file_name += f"_obj={result_object_name}"
    selected_metrics_file_name += f"_id={id}"

    
    if is_test_mode:
        selected_metrics_file_name += "_test_mode"
        print(f"Running test_inference_strategy_with_optim in test mode with reduced processing")
        
        
        if len(error_bounds) > 10:
            
            step = len(error_bounds) // 10
            error_bounds = error_bounds[::step]
            error_bounds_percentages = error_bounds_percentages[::step]
            print(f"Reduced error bounds from {len(error_bounds_percentages)} to {len(error_bounds)}")
            
        
        if step_size > 20:
            step_size = 20
            print(f"Limited step_size to {step_size}")

    selected_pdf_file_name = selected_metrics_file_name + ".pdf"
    selected_csv_file_name = selected_metrics_file_name + ".csv"
    csv_path = f"{os.path.join(parent_path_pics, selected_csv_file_name)}"
    
    
    
    
    

    before_start = DateUtils.now()

    if (use_other_project):
        raise ValueError("Other project is not supported yet with 'optim'")

    list_of_selected_model_metrics_objects = []

    error_bounds_count = len(error_bounds)
    for idx, selected_error_bound in enumerate(error_bounds):
        optim_lr_to_abs_e = np.abs(np.log10(inference_optim_lr))
        
        
        
        

        ebm_optim_energy_array = result_object_train[
            f"ebm_optim_energy_s{inference_steps}_e{int(optim_lr_to_abs_e)}"
        ]
        ebm_optim_mae_y_hat_array = result_object_train["mse_init_orig_model"]
        ebm_optim_mse_array = result_object_train[
            f"ebm_optim_mse_s{inference_steps}_e{int(optim_lr_to_abs_e)}"
        ]

        
        
        

        selected_error_bound_percentage = error_bounds_percentages[idx]

        

        global df_total_filtered

        
        
        
        ENERGY_KEY = "noisy_energies_y_hat"
        MSE_KEY = "noisy_mse_y_hat"
        MAE_KEY = "mae_from_sota_y_hat"

        df_total_filtered = pd.DataFrame(
            {
                ENERGY_KEY: (ebm_optim_energy_array),
                MSE_KEY: (ebm_optim_mse_array),
                MAE_KEY: (ebm_optim_mae_y_hat_array),
            }
        )

        global aggregate_df_data_df, aggregate_df_data_df_o

        
        
        aggregate_df_data_df, range_of_energies = calculate_energy_bounds(
            df_total_filtered=df_total_filtered,
            energy_key=ENERGY_KEY,
            mse_key=MSE_KEY,
            step_size=step_size,
        )

        
        
        
        
        
        

        def check_if_dataframes_are_equal():
            original = aggregate_df_data_df[
                ["mean_mse", "actual_count", "std", "count"]
            ]
            headed = aggregate_df_data_df_o.head(len(aggregate_df_data_df_o))[
                ["mean_mse", "actual_count", "std", "count"]
            ]
            dataframes_equal = original.round(3).equals(headed.round(3))
            if not dataframes_equal:
                raise ValueError("Dataframes are not equal")

        

        
        
        selection_criteria = "original"  
        (
            selected_aggregate_df_data_df,
            updated_aggregate_df_data_df,
        ) = calculate_aggregate_df(
            aggregate_df_data_df=aggregate_df_data_df,
            selected_error_bound=selected_error_bound,
            selection_criteria=selection_criteria,
        )

        
        selected_aggregate_df_data_df["sum_error"] = (
            selected_aggregate_df_data_df["mean_mse"]
            * selected_aggregate_df_data_df["actual_count"]
        )
        selected_mse = (
            selected_aggregate_df_data_df["sum_error"].sum()
            / selected_aggregate_df_data_df["actual_count"].sum()
        )
        train_coverage = (
            selected_aggregate_df_data_df["actual_count"].sum()
            / aggregate_df_data_df["actual_count"].sum()
        )

        
        

        noisy_energy_array = result_object_to_use[
            f"ebm_optim_energy_s{inference_steps}_e{int(optim_lr_to_abs_e)}"
        ].reshape(-1, 1)
        
        
        
        noisy_energies_to_test = noisy_energy_array

        df_total_orig_model_only_object = {
            "noisy_energies": np.concatenate(noisy_energies_to_test),
            "noisy_mse": np.concatenate(
                result_object_to_use["mse_init_orig_model"].reshape(-1, 1)
            ),
        }

        global df_total_orig_model_only
        df_total_orig_model_only = pd.DataFrame(df_total_orig_model_only_object)

        
        

        (
            selected_df_total_orig_model,
            filtered_df_total_orig_model,
        ) = calculate_selected_and_filtered_dfs(
            selected_aggregate_df_data_df=selected_aggregate_df_data_df,
            range_of_energies=range_of_energies,
            step_size=step_size,
        )

        
        

        
        common_select_result_obj = generate_select_result_obj(
            selected_df_total_orig_model=selected_df_total_orig_model,
            filtered_df_total_orig_model=filtered_df_total_orig_model,
            selected_error_bound=selected_error_bound,
        )
        select_result_obj = {
            **common_select_result_obj,
            "selected_error_bound_percentage": selected_error_bound_percentage,
            "selection_id": id,
            "n_percentages": len(error_bounds_percentages),
            "result_object_name": result_object_name,
            "train_coverage": train_coverage,
            "selection_criteria": "original",
        }

        list_of_selected_model_metrics_objects.append(select_result_obj)

    global selected_metrics_object_df
    selected_metrics_object_df = pd.DataFrame(list_of_selected_model_metrics_objects)

    if output_pdf:
        try:
            
            
            import seaborn as sns

            plt.figure(figsize=(12, 8), dpi=200)
            p = sns.barplot(
                x="energy_range",
                y="mean_mse",
                hue="count",
                data=aggregate_df_data_df,
                ci=None,
                dodge=False,  
                palette="coolwarm",
            )

            try:
                for container in p.containers:
                    p.bar_label(
                        container,
                        labels=aggregate_df_data_df["actual_count"],
                        label_type="center",
                        padding=-20,
                    )
            except:
                ...
                

            p.set_xlabel("Energy value [x; x+0.1)")
            p.set_ylabel("MSE of the sample")
            
            
            p.errorbar(
                x=range(len(aggregate_df_data_df["energy_range"])),
                y=aggregate_df_data_df["mean_mse"],
                yerr=aggregate_df_data_df["std"] / 2,
                fmt="none",
                c="black",
                capsize=5.0,
            )

            p.text(
                0.5,
                0.1,
                selected_metrics_file_name,
                size=8,
                weight="light",
                ha="center",
                va="center",
                transform=p.transAxes,
            )

            plt.legend(title="Sample count")
            plt.title(
                f"0. Noisy Energy correlation OPTIM barplot steps={inference_steps}, lr={inference_optim_lr}"
            )

            
            plt.savefig(f"{os.path.join(parent_path_pics, selected_pdf_file_name)}")

        except ValueError as e:
            print(f"Error while saving selected_pdf file. e={e}")

    try:
        csv_path = f"{os.path.join(parent_path_pics, selected_csv_file_name)}"
        selected_metrics_object_df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error while saving selected_csv file. e={e}")

    return selected_metrics_object_df


def perform_selective_inference_experiments_with_optim(
    result_object: dict,  
    result_object_train: dict,
    result_object_test: dict,
    
    parent_path_pics: str,
    is_different_project=False,
    is_test_mode=False,
    inference_steps_custom=None,
    inference_optim_lr_custom=None,
):
    """
    Run selective inference experiments with optimization-based strategy.
    
    This function will run exactly once with the specified parameters and return the result dataframe.
    """

    try:
        before_visualization = DateUtils.now()
        val_error = result_object_train["mse_init_orig_model"].mean()
        val_error_orig = result_object_train["mse_orig"].mean()

        if val_error - val_error_orig > 0.05:
            print(
                f">>>\n>>>\nWARNING: ERROR MISMATCH!\nebm={val_error} != orig={val_error_orig}>>>\n>>>"
            )

        error_bounds_percentages = [0.75 + i * 0.05 for i in range(80)]
        error_bounds = list(map(lambda per: per * val_error, error_bounds_percentages))

        
        if inference_steps_custom is None:
            inference_steps_custom = 25  
        
        if inference_optim_lr_custom is None:
            inference_optim_lr_custom = 0.01  
            
        
        if is_test_mode:
            print("Running in test mode - using reduced inference steps")
            inference_steps_custom = min(inference_steps_custom, 10)  
            
        print(f"Running optimization-based inference with: steps={inference_steps_custom}, lr={inference_optim_lr_custom}")

        result_object_to_use = result_object
        mse_init_orig_model = result_object_to_use["mse_init_orig_model"].mean()

        idx = 0
        before_execution = DateUtils.now()

        
        val_optim_selected_metrics_object_df = test_inference_strategy_with_optim(
            inference_steps=inference_steps_custom,
            inference_optim_lr=inference_optim_lr_custom,
            result_object_train=result_object_train,
            parent_path_pics=parent_path_pics,
            id=idx,
            result_object_to_use=result_object,
            result_object_name="val",
            error_bounds=error_bounds,
            error_bounds_percentages=error_bounds_percentages,
            is_test_mode=is_test_mode,
        )
        
        
        test_optim_selected_metrics_object_df = test_inference_strategy_with_optim(
            inference_steps=inference_steps_custom,
            inference_optim_lr=inference_optim_lr_custom,
            result_object_train=result_object_train,
            parent_path_pics=parent_path_pics,
            id=idx,
            result_object_to_use=result_object_test,
            result_object_name="test",
            error_bounds=error_bounds,
            error_bounds_percentages=error_bounds_percentages,
            is_test_mode=is_test_mode,
        )

        
        
        target_coverages = [0.1, 0.3, 0.5, 0.7, 0.9]  
        
        
        filtered_val_df = pd.DataFrame()
        filtered_test_df = pd.DataFrame()
        
        for target in target_coverages:
            
            if not val_optim_selected_metrics_object_df.empty:
                
                val_optim_selected_metrics_object_df['coverage_diff'] = abs(val_optim_selected_metrics_object_df['train_coverage'] - target)
                
                closest_row_val = val_optim_selected_metrics_object_df.loc[val_optim_selected_metrics_object_df['coverage_diff'].idxmin()].copy()
                
                filtered_val_df = pd.concat([filtered_val_df, pd.DataFrame([closest_row_val])], ignore_index=True)
            
            
            if not test_optim_selected_metrics_object_df.empty:
                
                test_optim_selected_metrics_object_df['coverage_diff'] = abs(test_optim_selected_metrics_object_df['train_coverage'] - target)
                
                closest_row_test = test_optim_selected_metrics_object_df.loc[test_optim_selected_metrics_object_df['coverage_diff'].idxmin()].copy()
                
                filtered_test_df = pd.concat([filtered_test_df, pd.DataFrame([closest_row_test])], ignore_index=True)
        
        
        if 'coverage_diff' in filtered_val_df.columns:
            filtered_val_df = filtered_val_df.drop('coverage_diff', axis=1)
        if 'coverage_diff' in filtered_test_df.columns:
            filtered_test_df = filtered_test_df.drop('coverage_diff', axis=1)
            
        print(f"Filtered to {len(filtered_val_df)} rows with target coverages: {target_coverages}")
        
        return filtered_val_df, filtered_test_df, result_object
        
    except Exception as e:
        raise e
        return None, None, None
