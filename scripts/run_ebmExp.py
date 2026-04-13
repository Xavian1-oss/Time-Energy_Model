import argparse
import random
import sys
from pathlib import Path

# Running as `python scripts/run_ebmExp.py` puts `scripts/` on sys.path first;
# project modules (analysis, exp, ...) live at the repository root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import zipfile

import numpy as np
import pandas as pd
import torch

import analysis
from utils.graph_energy_gate import run_graph_gate_evaluation
from data_provider.experiment_data import ExperimentData
from exp.exp_main_energy import Exp_Main_Energy
from run_commons import TorchDeviceUtils, ExperimentConstants
from torch_utils import unwrap_dataparallel
from utilz import *


def get_default_args():
    """
    Returns a dictionary of default arguments for the experiment.
    These are the hardcoded values that won't be exposed to the user.
    """
    return {
        
        "is_training": 1,
        "model_id": "v2_ETTh1_PatchTST_seq_len_96",
        "name": "neoebm_experiment",
        # By default we do not attach an adaptive graph head; this
        # will be enabled automatically for multivariate (M) tasks
        # after parsing CLI args.
        "use_adaptive_graph": False,
        # When use_adaptive_graph: Dirichlet graph regularization vs ground-truth graph (see exp_main_energy).
        "gate_graph_loss_weight": 0.1,
        # When use_adaptive_graph: extra term to match graph energy on preds vs MSE (see graph_energy_gate).
        "gate_graph_align_weight": 0.05,
        "gate_graph_align_log_mse": True,
        
        "data": "custom",  
        "site_id": "None",
        "target_site_id": "None",
        "target": "OT",
        "root_path": os.path.join(os.getcwd(), "dataset"),
        
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 48,
        "embed_type": 0,
                
        "c_out": 1,
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 32,
        "moving_avg": 25,
        "factor": 1,
        "dropout": 0.05,
        "embed": "timeF",
        "activation": "gelu",
        "distil": True,
        "output_attention": False,
        "do_predict": False,
        "individual": False,
        
        "num_workers": 0,
        "itr": 1,
        "train_epochs": 30,
        "batch_size": 8,
        "patience": 3,
        "learning_rate": 0.001,
        "des": "experiment",
        "loss": "mse",
        "lradj": "type1",
        "use_amp": False,
                
        "ebm_epochs": 30,
        "ebm_samples": 256,
        "ebm_hidden_size": 16,
        "ebm_num_layers": 1,
        "ebm_decoder_num_layers": 4,
        "ebm_predictor_size": 96,
        "ebm_decoder_size": 96,
        "ebm_inference_optim_lr": 0.001,
        "ebm_inference_optim_steps": 50,
        "ebm_inference_batch_size": 32,
        "ebm_validate_during_training_step": 10,
        "ebm_seed": 2024,
        "ebm_training_strategy": "train_y_and_xy_together",
        "ebm_cd_step_size": 0.1,
        "ebm_cd_num_steps": 10,
        "ebm_cd_alpha": 0.9,
        "ebm_cd_sched_rate": 1.0,
        "ebm_margin_loss": 0.0,
        "ebm_training_method": "nce",
        "ebm_optim_lr": 1e-3,
        
        "only_rerun_inference": 0,
        "should_log": True,
        "force_retrain_xy_dec": False,
        "force_retrain_orig_model": False,
        "force_retrain_y_enc": False,
        "only_output_model_params": 0,
        "is_test_mode": 0,
        
        "use_gpu": True,
        "gpu": 0,
        "use_multi_gpu": False,
        "devices": "0,1,2,3",
        "test_flop": False,
        
        "experiment_only_on_given_model_path": "None",
        "checkpoints": "./checkpoints/",
        
        "noisy_std": 0.1,
        "inference_steps": 25,
        "inference_optim_lr": 0.01,
        
        "version": "Fourier",
        "mode_select": "random",
        "modes": 64,
        
        "top_k": 5,
        "num_kernels": 6,
    }

def get_model_specific_configs(model_name):
    """
    Returns model-specific configurations based on the selected model.
    """
    configs = {
        "TimesNet": {
            "ebm_model_name": "times_net_mlp_concat",
            "top_k": 5,
            "num_kernels": 6,
            
            "batch_size": 64,           
            "learning_rate": 0.001,     
            "d_model": 16,              
            "d_ff": 32                  
        },
        "Autoformer": {
            "ebm_model_name": "autoformer_mlp_concat"
        },
        "Informer": {
            "ebm_model_name": "informer_mlp_concat",  
        },
        "PatchTST": {
            "ebm_model_name": "patchtst_mlp_concat"
        },
        "FEDformer": {
            "ebm_model_name": "fedformer_mlp_concat",
            "version": "Fourier",
            "mode_select": "random",
            "modes": 64
        }
    }
    
    
    return configs.get(model_name, {"ebm_model_name": "mlp_concat"})

def get_dataset_specific_configs(data_path):
    """
    Returns dataset-specific configurations based on the data_path.
    Also determines the 'data' parameter.
    """
    
    config = {
        "data": "custom",
        "freq": "h"
    }
    
    
    if data_path.startswith("ETTh1"):
        config["data"] = "ETTh1"
    elif data_path.startswith("ETTh2"):
        config["data"] = "ETTh2"
    elif data_path.startswith("ETTm1"):
        config["data"] = "ETTm1"
    elif data_path.startswith("ETTm2"):
        config["data"] = "ETTm2"
    elif data_path.startswith("exchange_rate"):
        config["data"] = "custom"
        config["enc_in"] = 8
        config["dec_in"] = 8
    elif data_path.startswith("electricity"):
        config["data"] = "custom"
        config["enc_in"] = 321
        config["dec_in"] = 321
        # Use a smaller batch size for high-dimensional electricity dataset
        config["batch_size"] = 2
    elif data_path.startswith("traffic"):
        config["data"] = "custom"
        config["enc_in"] = 862
        config["dec_in"] = 862
        # Use an even smaller batch size for high-dimensional traffic dataset
        config["batch_size"] = 1
    elif data_path.startswith("weather"):
        config["data"] = "custom"
        config["enc_in"] = 21
        config["dec_in"] = 21
    elif data_path.startswith("national_illness"):
        config["data"] = "custom"
        config["enc_in"] = 7
        config["dec_in"] = 7

    return config

def create_simplified_parser():
    """
    Creates a simplified argument parser with only the selectable parameters.
    """
    parser = argparse.ArgumentParser(
        description="Simplified NeoEBM Time Series Forecasting"
    )
    
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["TimesNet", "Autoformer", "Informer", "FEDformer", "PatchTST"],
        help="Model architecture to use"
    )
    
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Data file to use (e.g., exchange_rate.csv, ETTh1.csv)"
    )
    
    
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        choices=["S", "MS", "M"],
        help="Forecasting task: S (univariate), MS (multivariate->univariate), M (multivariate)"
    )
    
    
    parser.add_argument(
        "--inference_strategy",
        type=str,
        required=False,
        default="noise",
        choices=["noise", "optim"],
        help="Inference strategy: noise (default) or optim"
    )
    
    
    parser.add_argument(
        "--noisy_std",
        type=float,
        required=False,
        help="Standard deviation for noise-based inference (default: 0.1)"
    )
    
    
    parser.add_argument(
        "--inference_steps",
        type=int,
        required=False,
        help="Number of steps for optimization-based inference (default: 25)"
    )
    
    parser.add_argument(
        "--inference_optim_lr",
        type=float,
        required=False,
        help="Learning rate for optimization-based inference (default: 0.01)"
    )
    
    
    parser.add_argument(
        "--output_parent_path",
        type=str,
        required=True,
        help="Output directory for experiment results"
    )
    
    
    parser.add_argument(
        "--is_test_mode",
        type=int,
        required=False,
        default=1,
        choices=[0, 1],
        help="Enable test mode (1) or full mode (0). Test mode runs with fewer iterations for faster testing."
    )

    # Optional control over graph usage. "auto" (default) keeps the
    # current behaviour: on multivariate (M) tasks we train an
    # adaptive graph head and run graph-based selective inference.
    # "none" disables both adaptive graph training and graph-gate
    # evaluation so that only TEM/EBM selective inference is run.
    parser.add_argument(
        "--graph_mode",
        type=str,
        required=False,
        default="auto",
        choices=["auto", "none"],
        help="Graph usage: auto (default) or none (TEM-only on M tasks).",
    )

    # Optional control over EBM usage. "auto" (default) keeps the
    # current behaviour: train the EBM head and run TEM-based
    # selective inference. "none" disables both EBM training and
    # EBM/TEM analysis so that only the forecasting backbone and
    # (optionally) graph-based selective inference are run.
    parser.add_argument(
        "--ebm_mode",
        type=str,
        required=False,
        default="auto",
        choices=["auto", "none"],
        help="EBM usage: auto (default, train + analyze EBM) or none (skip EBM, graph-only).",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=False,
        default=None,
        help="Override auto-derived checkpoint id (default embeds dataset + model + seq_len).",
    )

    return parser


parser = create_simplified_parser()
user_args = parser.parse_args()


default_args = get_default_args()


model_configs = get_model_specific_configs(user_args.model)
default_args.update(model_configs)


dataset_configs = get_dataset_specific_configs(user_args.data_path)
default_args.update(dataset_configs)


args = argparse.Namespace()


for key, value in default_args.items():
    setattr(args, key, value)


for key, value in vars(user_args).items():
    if value is not None:  
        setattr(args, key, value)


args.test_strategy = args.inference_strategy
delattr(args, 'inference_strategy')


if args.features == "S":
    args.enc_in = 1
    args.dec_in = 1
elif not hasattr(args, 'enc_in') or not hasattr(args, 'dec_in'):
    # 对 ETT 系列数据，如果是多变量输入（M 或 MS），默认通道数为 7
    if args.data in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"] and args.features in ["M", "MS"]:
        args.enc_in = 7
        args.dec_in = 7


# Decide whether to enable adaptive graph training. By default
# (graph_mode == "auto"), we enable it for multivariate (M) tasks so
# that the learnable dependency graph head is trained jointly with the
# backbone. When graph_mode == "none", we explicitly disable it so
# that only TEM/EBM runs even on M tasks.
graph_mode = getattr(args, "graph_mode", "auto")
if graph_mode == "none":
    # 强制关闭自适应图，无论默认值如何
    args.use_adaptive_graph = False
else:
    # 对 auto 模式，显式根据特征类型覆盖默认值：M 任务开启图头，其它关闭。
    args.use_adaptive_graph = (args.features == "M")


# Multivariate (M): output dim must match channels for backbone projections
# (e.g. Autoformer/Informer/FEDformer use configs.c_out; PatchTST uses enc_in in the head).
if args.features == "M" and hasattr(args, "enc_in") and args.enc_in is not None:
    args.c_out = args.enc_in

# Readable, unique checkpoint token per (dataset, model, seq_len); old default was ETTh1/PatchTST-specific.
if user_args.model_id is not None:
    args.model_id = user_args.model_id
else:
    _data_tag = (
        Path(args.data_path).stem.replace(".", "_")
        if getattr(args, "data", "") == "custom"
        else str(args.data).replace(".", "_")
    )
    args.model_id = f"v2_{_data_tag}_{args.model}_sl{args.seq_len}"


if args.only_rerun_inference:
    print(f">> ONLY RE-RUNNING INFERENCE!")


random.seed(args.ebm_seed)
torch.manual_seed(args.ebm_seed)
np.random.seed(args.ebm_seed)



args.use_gpu, mps_available = TorchDeviceUtils.check_if_should_use_gpu(args)

if args.use_gpu and args.use_multi_gpu and not mps_available:
    args.devices = args.devices.replace(" ", "")
    device_ids = args.devices.split(",")
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print("---\n")
print("Args in experiment:")
print(args)


def verify_args(args):
    """
    Raises exception if args are incompatible
    """
    if args.features == "S":
        if args.enc_in != 1 or args.dec_in != 1:
            raise ValueError(
                f"Given 'features' == 'S', enc_in and dec_in can only be 1! (Currently dec_in: {args.dec_in}, enc_in: {args.enc_in})"
            )

    if args.features == "MS":
        if args.enc_in == 1 and args.dec_in == 1:
            raise ValueError(
                f"Given 'features' == 'MS', enc_in and dec_in CANNOT BE 1! (Currently dec_in: {args.dec_in}, enc_in: {args.enc_in})"
            )

    if hasattr(args, 'feature') and args.feature is not None:
        raise ValueError(
            f"'--feature' argument is not used in this script. Use '--features' instead!"
        )

    return True


def save_learned_graph_adjacency(
    exp: Exp_Main_Energy,
    full_ebm_path: str,
    only_if_missing: bool = False,
) -> None:
    """Persist learned adjacency next to checkpoints (including ebm_mode='none')."""
    run_args = exp.args
    if run_args.features != "M" or getattr(run_args, "graph_mode", "auto") == "none":
        return
    backbone = unwrap_dataparallel(exp.model)
    if not hasattr(backbone, "dep_graph_builder") or backbone.dep_graph_builder is None:
        return
    graph_save_path = Path(full_ebm_path).parent / "learned_graph_A.npy"
    if only_if_missing and graph_save_path.exists():
        return
    try:
        A = backbone.dep_graph_builder()
        if hasattr(A, "detach"):
            A_np = A.detach().cpu().numpy()
        else:
            A_np = np.asarray(A)
        np.save(graph_save_path, A_np)
        print(f"[GraphEnergy] Saved learned adjacency matrix to {graph_save_path}")
    except Exception as e:
        print(
            f"[GraphEnergy][WARN] Failed to save learned adjacency matrix, continuing without it: {e}"
        )


def write_run_summary(run_args, setting: str, full_ebm_path: str) -> None:
    out_parent = getattr(run_args, "output_parent_path", None)
    if out_parent in (None, "", "None"):
        return
    try:
        out_dir = Path(str(out_parent))
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = Path(full_ebm_path).parent
        analysis_dir = ckpt_dir / f"local_pics_{run_args.data}"
        summary_path = out_dir / "run_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"setting: {setting}\n")
            f.write(f"full_ebm_path: {full_ebm_path}\n")
            f.write(f"checkpoint_dir: {ckpt_dir}\n")
            f.write(f"analysis_dir: {analysis_dir}\n")
        print(f"Wrote run summary to {summary_path}")
    except Exception as e:
        print(
            f"[WARN] Failed to write summary under output_parent_path={out_parent}: {e}"
        )


Exp = Exp_Main_Energy


def run_analysis(
    experiment_data: ExperimentData,
    path_to_given_model=args.experiment_only_on_given_model_path,
    override_train_experiment_data=None,
    ignore_cached_result_objects=False,
    ignore_cached_result_objects_except_train=False,
    parent_path_pics_folder_postfix=args.data,
):
    print(">>> RUNNING ANALYSIS >>>")
    before_analyis = DateUtils.now()

    from pathlib import Path

    path_to_given_model_path_object = Path(path_to_given_model)
    model_name = "_".join(
        list(map(lambda s: s.strip("'"), path_to_given_model_path_object.parts[1:]))
    ).split(".")[0]
    if not path_to_given_model_path_object.exists():
        raise ValueError(f"File with path '{path_to_given_model}' does not exist!")

    global parent_path_pics
    parent_path_pics = os.path.join(
        path_to_given_model_path_object.parent,
        f"local_pics_{parent_path_pics_folder_postfix}",
    )
    FileUtils.create_dir(parent_path_pics)
    print(f">>> Analysis path: {parent_path_pics}")

    before_test = DateUtils.now()

    serialized_result_object_path = os.path.join(
        path_to_given_model_path_object.parent, "result_objects"
    )
    FileUtils.create_dir(str(serialized_result_object_path))

    def generate_cache_path(data_loader_name, exp_data: ExperimentData):
        return Path(
            os.path.join(
                serialized_result_object_path,
                f"result_obj_{FileUtils.sanitize_filename(exp_data.short_id)}_{data_loader_name}_{ExperimentConstants.RESULT_OBJECT_VERSION_POSTFIX}.npz",
            )
        )

    def load_or_generate_result_object(
        cache_path: Path,
        data_loader_name: str,
        experiment_data: ExperimentData,
        override_data: ExperimentData = None,
        ignore_cached_result_objects: bool = ignore_cached_result_objects,
        ignore_cached_result_objects_except_train: bool = ignore_cached_result_objects_except_train,
    ):
        ignore_cache = ignore_cached_result_objects or (
            ignore_cached_result_objects_except_train and data_loader_name != "train"
        )
        if not ignore_cache and cache_path.exists():
            try:
                result_object = dict(np.load(str(cache_path), allow_pickle=True))
            except (OSError, ValueError, zipfile.BadZipFile) as e:
                # Common after disk-full or killed mid-write: truncated npz / bad ZIP.
                print(
                    f"[Cache] Unreadable result cache (will delete and regenerate): "
                    f"{cache_path}\n  ({type(e).__name__}: {e})"
                )
                try:
                    cache_path.unlink()
                except OSError:
                    pass
            else:
                dataset = getattr(experiment_data, f"{data_loader_name}_data")
                return result_object, dataset

        data_to_use = (
            override_data if override_data is not None else experiment_data
        )
        result_object, dataset = exp.test_adhoc_energy(
            path_to_given_model,
            experiment_data=data_to_use,
            # 简化版统计不再依赖噪声采样结果，这里关闭宽泛噪声实验以大幅提速
            do_run_wide_experiments=False,
            data_loader_name=data_loader_name,
        )
        np.savez(cache_path, **result_object)
        return result_object, dataset

    train_cache_path = generate_cache_path(
        "train",
        override_train_experiment_data or experiment_data,
    )
    result_object_train, train_dataset = load_or_generate_result_object(
        train_cache_path,
        "train",
        experiment_data,
        override_data=override_train_experiment_data,
        ignore_cached_result_objects=ignore_cached_result_objects,
        ignore_cached_result_objects_except_train=ignore_cached_result_objects_except_train,
    )

    val_cache_path = generate_cache_path("val", experiment_data)
    result_object, val_dataset = load_or_generate_result_object(
        val_cache_path,
        "val",
        experiment_data,
        ignore_cached_result_objects=ignore_cached_result_objects,
        ignore_cached_result_objects_except_train=ignore_cached_result_objects_except_train,
    )

    test_cache_path = generate_cache_path("test", experiment_data)
    result_object_test, test_dataset = load_or_generate_result_object(
        test_cache_path,
        "test",
        experiment_data,
        ignore_cached_result_objects=ignore_cached_result_objects,
        ignore_cached_result_objects_except_train=ignore_cached_result_objects_except_train,
    )

    
    metatitle = str(result_object["metatitle"]) + "_ground"

    
    test_strategy = "noise"  
    if hasattr(args, 'test_strategy'):
        test_strategy = args.test_strategy
    
    print(f"Running inference strategy: {test_strategy}")
    
    if test_strategy == "noise":
        
        print("Running noise-based inference strategy")
        val_metrics_df, test_metrics_df, output_1 = analysis.perform_selective_inference_experiments(
            result_object,
            result_object_train,
            result_object_test,
            parent_path_pics,
            train_dataset_scaler=train_dataset.scaler,
            is_different_project=False,
            is_test_mode=args.is_test_mode is not None and args.is_test_mode == 1,
            noisy_std_custom=args.noisy_std if hasattr(args, 'noisy_std') else None,
        )
        
        if val_metrics_df is not None:
            print(f"Noise-based inference completed successfully.")
            
            
            val_metrics_path = os.path.join(parent_path_pics, "noise_val_metrics_filtered.csv")
            test_metrics_path = os.path.join(parent_path_pics, "noise_test_metrics_filtered.csv")
            val_metrics_df.to_csv(val_metrics_path, index=False)
            test_metrics_df.to_csv(test_metrics_path, index=False)
            print(f"Saved filtered metrics to {val_metrics_path} and {test_metrics_path}")
            
        else:
            print("Noise-based inference did not return valid results.")
            
    elif test_strategy == "optim":
        
        print("Running optimization-based inference strategy")
        val_metrics_df, test_metrics_df, output_1 = analysis.perform_selective_inference_experiments_with_optim(
            result_object,
            result_object_train,
            result_object_test,
            parent_path_pics,
            is_different_project=False,
            is_test_mode=args.is_test_mode is not None and args.is_test_mode == 1,
            inference_steps_custom=args.inference_steps if hasattr(args, 'inference_steps') else None,
            inference_optim_lr_custom=args.inference_optim_lr if hasattr(args, 'inference_optim_lr') else None,
        )

        if val_metrics_df is not None:
            print(f"Optimization-based inference completed successfully.")
            
            
            val_metrics_path = os.path.join(parent_path_pics, "optim_val_metrics_filtered.csv")
            test_metrics_path = os.path.join(parent_path_pics, "optim_test_metrics_filtered.csv")
            val_metrics_df.to_csv(val_metrics_path, index=False)
            test_metrics_df.to_csv(test_metrics_path, index=False)
            print(f"Saved filtered metrics to {val_metrics_path} and {test_metrics_path}")
            
        else:
            print("Optimization-based inference did not return valid results.")
            
    else:
        print(f"Unknown test strategy: {test_strategy}")
        output_1 = None
        test_metrics_df = None
        
    
    if test_metrics_df is not None:
        print("\n" + "="*80)
        print(f"FINAL FILTERED TEST METRICS FOR {test_strategy.upper()} STRATEGY:")
        print("="*80)
        pd.set_option('display.max_columns', None)  
        pd.set_option('display.width', 1000)        
        pd.set_option('display.precision', 4)       
        print(test_metrics_df)
        print("="*80 + "\n")
        
    return result_object, result_object_train, result_object_test, train_dataset, test_metrics_df


if args.is_training:
    for ii in range(args.itr):
        
        setting = "{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
            ExperimentConstants.SETTINGS_PREFIX,
            args.model_id,
            args.model,
            args.data,
            args.data_path.replace(
                ".", "_"
            ),  
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )
        if args.site_id != "None":
            site_id_string = args.site_id.replace(",", "_")
            setting = setting + f"_{site_id_string}"

        verify_args(args)

        # Print test mode status
        if args.is_test_mode == 1:
            print(f"Running in TEST MODE - using reduced epochs and parameters")
        
        exp = Exp(args, setting=setting)  

        ebm_mode = getattr(args, "ebm_mode", "auto")

        # Decide whether we can reuse existing checkpoints (backbone + EBM)
        # instead of retraining from scratch. This is enabled when both the
        # forecasting checkpoint and full_ebm.pth exist and no force_retrain
        # flags are set.
        full_ebm_path = exp.get_full_ebm_path(setting)
        model_ckpt_path = exp._model_checkpoint_path(setting)
        if ebm_mode == "none":
            # In graph-only mode we never train or use the EBM, so
            # warm-start only depends on the forecasting checkpoint.
            can_reuse_checkpoints = (
                Path(model_ckpt_path).exists()
                and not getattr(args, "force_retrain_orig_model", False)
            )
        else:
            can_reuse_checkpoints = (
                Path(full_ebm_path).exists()
                and Path(model_ckpt_path).exists()
                and not getattr(args, "force_retrain_orig_model", False)
                and not getattr(args, "force_retrain_y_enc", False)
                and not getattr(args, "force_retrain_xy_dec", False)
            )

        if not args.only_rerun_inference and not can_reuse_checkpoints:
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            before_exp_train = DateUtils.now()

            experiment_data = ExperimentData.from_args(args)

            exp.train(setting, experiment_data=experiment_data)
            after_exp_train = DateUtils.now()
            print(f"Training completed in {after_exp_train - before_exp_train}")

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting, experiment_data=experiment_data)

            if args.do_predict:
                print(
                    ">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                        setting
                    )
                )
                exp.predict(setting, True)

            if ebm_mode != "none":
                full_ebm, full_ebm_path = exp.train_energy(
                    setting, experiment_data=experiment_data
                )
            else:
                # In graph-only mode, we still define full_ebm_path so that
                # downstream paths (e.g., for graph metrics) use the same
                # directory structure, but we never actually train or save
                # an EBM.
                full_ebm_path = exp.get_full_ebm_path(setting)

            save_learned_graph_adjacency(exp, full_ebm_path, only_if_missing=False)

            # For multivariate forecasting tasks (features == "M"), run
            # graph-structural selective inference in addition to (or, when
            # ebm_mode == "none", instead of) TEM/EBM analysis. This
            # produces graph_val_metrics_filtered.csv and
            # graph_test_metrics_filtered.csv.
            if args.features == "M" and getattr(args, "graph_mode", "auto") != "none":
                try:
                    from pathlib import Path

                    full_ebm_path_obj = Path(full_ebm_path)
                    parent_dir = full_ebm_path_obj.parent
                    parent_path_pics_graph = os.path.join(
                        parent_dir,
                        f"local_pics_{args.data}",
                    )
                    FileUtils.create_dir(parent_path_pics_graph)

                    print(
                        "[GraphEnergyGate] Running graph-based selective inference for multivariate task..."
                    )
                    run_graph_gate_evaluation(
                        exp=exp,
                        experiment_data=experiment_data,
                        parent_path_pics=parent_path_pics_graph,
                    )
                except Exception as e:
                    print(
                        f"[GraphEnergyGate][WARN] Graph-based evaluation failed and will be skipped: {e}"
                    )
        elif not args.only_rerun_inference and can_reuse_checkpoints:
            # Warm-start path: reuse existing trained backbone + EBM without
            # retraining. We only need to load the forecasting checkpoint so
            # that online graph-gate evaluation can use the learned
            # dependency graph; TEM/EBM analysis will read full_ebm.pth
            # directly from disk.
            print(
                f"[WarmStart] Found existing checkpoints for setting='{setting}'. "
                f"Reusing trained backbone and EBM without retraining."
            )

            device = torch.device("cuda" if args.use_gpu else "cpu")
            try:
                exp.model.load_state_dict(
                    torch.load(model_ckpt_path, map_location=device)
                )
            except Exception as e:
                print(
                    f"[WarmStart][WARN] Failed to load backbone checkpoint from {model_ckpt_path}: {e}. "
                    f"Falling back to full training."
                )
                # Fall back to the original training branch
                print(
                    ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
                )
                before_exp_train = DateUtils.now()

                experiment_data = ExperimentData.from_args(args)

                exp.train(setting, experiment_data=experiment_data)
                after_exp_train = DateUtils.now()
                print(f"Training completed in {after_exp_train - before_exp_train}")

                print(
                    ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
                )
                exp.test(setting, experiment_data=experiment_data)

                if args.do_predict:
                    print(
                        ">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                            setting
                        )
                    )
                    exp.predict(setting, True)

                if ebm_mode != "none":
                    full_ebm, full_ebm_path = exp.train_energy(
                        setting, experiment_data=experiment_data
                    )
                else:
                    full_ebm_path = exp.get_full_ebm_path(setting)

                save_learned_graph_adjacency(exp, full_ebm_path, only_if_missing=False)

                if args.features == "M" and getattr(args, "graph_mode", "auto") != "none":
                    try:
                        from pathlib import Path

                        full_ebm_path_obj = Path(full_ebm_path)
                        parent_dir = full_ebm_path_obj.parent
                        parent_path_pics_graph = os.path.join(
                            parent_dir,
                            f"local_pics_{args.data}",
                        )
                        FileUtils.create_dir(parent_path_pics_graph)

                        print(
                            "[GraphEnergyGate] Running graph-based selective inference for multivariate task..."
                        )
                        run_graph_gate_evaluation(
                            exp=exp,
                            experiment_data=experiment_data,
                            parent_path_pics=parent_path_pics_graph,
                        )
                    except Exception as e2:
                        print(
                            f"[GraphEnergyGate][WARN] Graph-based evaluation failed and will be skipped: {e2}"
                        )
            else:
                # Successfully loaded backbone; optionally refresh adjacency
                # if missing (including graph-only / ebm_mode='none' runs).
                save_learned_graph_adjacency(exp, full_ebm_path, only_if_missing=True)

                # We still run graph-gate evaluation so that the current
                # run has fresh selective metrics if desired.
                experiment_data = ExperimentData.from_args(args)
                if args.features == "M" and getattr(args, "graph_mode", "auto") != "none":
                    try:
                        from pathlib import Path

                        full_ebm_path_obj = Path(full_ebm_path)
                        parent_dir = full_ebm_path_obj.parent
                        parent_path_pics_graph = os.path.join(
                            parent_dir,
                            f"local_pics_{args.data}",
                        )
                        FileUtils.create_dir(parent_path_pics_graph)

                        print(
                            "[GraphEnergyGate] Running graph-based selective inference for multivariate task (warm start)..."
                        )
                        run_graph_gate_evaluation(
                            exp=exp,
                            experiment_data=experiment_data,
                            parent_path_pics=parent_path_pics_graph,
                        )
                    except Exception as e:
                        print(
                            f"[GraphEnergyGate][WARN] Graph-based evaluation failed on warm start and will be skipped: {e}"
                        )
        else:
            full_ebm_path = exp.get_full_ebm_path(setting)
            if ebm_mode != "none" and not Path(full_ebm_path).exists():
                raise ValueError(
                    f"EARLY EXIT, ebm_path='{full_ebm_path}' does not exist!"
                )

        

        import os
        from pathlib import Path

        # Always persist args.csv next to the (potential) EBM path so that
        # downstream tooling can inspect the configuration, even when
        # ebm_mode == 'none' and no full_ebm.pth is saved.
        try:
            args_df = pd.DataFrame(vars(args), index=[0])
            args_df_path = os.path.join(Path(full_ebm_path).parent, "args.csv")
            args_df.to_csv(args_df_path)
        except Exception as e:
            print(f"Exception: {e}")

        write_run_summary(args, setting, full_ebm_path)

        if args.only_output_model_params:
            raise ValueError("NOT IMPLEMENTED!")

        # When ebm_mode == 'none', we explicitly skip TEM/EBM analysis and
        # final selective summaries; this run is intended to evaluate only
        # the forecasting backbone and graph-based gate.
        if ebm_mode == "none":
            print("[EBM] ebm_mode='none': skipping EBM training and TEM analysis; graph-only metrics have been saved.")
        else:
            before_analysis = DateUtils.now()
            experiment_data = ExperimentData.from_args(args)
            (
                result_object,
                result_object_train,
                result_object_test,
                train_dataset,
                test_metrics_df,
            ) = run_analysis(
                experiment_data=experiment_data,
                path_to_given_model=full_ebm_path,
                
            )

            after_analysis = DateUtils.now()
            print(f"Analysis completed in {after_analysis - before_analysis}")

            if test_metrics_df is not None:
                print("\n" + "="*80)
                print(f"FINAL RESULTS SUMMARY:")
                print("="*80)
                print(f"Strategy: {args.test_strategy if hasattr(args, 'test_strategy') else 'noise'}")
                if args.test_strategy == 'noise' and hasattr(args, 'noisy_std'):
                    print(f"Noise std: {args.noisy_std}")
                elif args.test_strategy == 'optim':
                    print(f"Inference steps: {args.inference_steps if hasattr(args, 'inference_steps') else 25}")
                    print(f"Inference optim lr: {args.inference_optim_lr if hasattr(args, 'inference_optim_lr') else 0.01}")
                
                
                print("\nKey metrics by coverage level:")
                print("-" * 100)
                print(f"{'Target Coverage':<15} {'Empirical Coverage':<20} {'Empirical Risk':<20} {'MSE Original':<20} {'Error Reduction (%)':<20}")
                print("-" * 100)
                
                
                target_coverages = [0.5, 0.6, 0.7, 0.8, 0.9]
                
                for i, row in test_metrics_df.iterrows():
                    target_coverage = target_coverages[i] if i < len(target_coverages) else None
                    empirical_coverage = row['train_coverage']
                    mse_selected = row['split_mse_selected']
                    mse_orig = row['split_mse_orig']
                    error_reduction = (1 - (mse_selected / mse_orig)) * 100 if mse_orig != 0 else 0
                    
                    print(f"{target_coverage:<15.2f} {empirical_coverage:<20.4f} {mse_selected:<20.4f} {mse_orig:<20.4f} {error_reduction:<20.2f}")
                
                print("="*100 + "\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

elif args.experiment_only_on_given_model_path != "None":
    experiment_data = ExperimentData.from_args(args)
    (
        result_object,
        result_object_train,
        result_object_test,
        train_dataset,
        test_metrics_df,
    ) = run_analysis(experiment_data)

print("Gotcha!")