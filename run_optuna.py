import click
import optuna
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial

from copy import deepcopy

from recbole.quick_start import run_recbole

from pathlib import Path
from omegaconf import open_dict


CONFIG_DIR = "configs"
OPTUNA_DIR = "optuna_outputs"

TARGET_METRIC = "ndcg@10"

def suggest_gsau_seq_cfg(config: DictConfig, trial: Trial) -> DictConfig:
    new_config = deepcopy(config)

    # new_config.n_layers = trial.suggest_int("n_layers", 1, 3, 4)
    new_config.n_layers = trial.suggest_int("n_layers", 1, 3)  
    new_config.n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])

    new_config.hidden_size = trial.suggest_categorical(
        "hidden_size", [32, 64, 128, 256]
    )

    new_config.embedding_size = new_config.hidden_size
    new_config.inner_size = trial.suggest_categorical(
        "inner_size", [128, 256, 512]
    )

    new_config.hidden_dropout_prob = trial.suggest_float(
        "hidden_dropout_prob", 0.0, 0.7, step=0.1
    )
    new_config.attn_dropout_prob = trial.suggest_float(
        "attn_dropout_prob", 0.0, 0.7, step=0.1
    )

    new_config.hidden_act = trial.suggest_categorical(
        "hidden_act", ["gelu", "relu", "swish"]
    )
    new_config.layer_norm_eps = trial.suggest_categorical(
        "layer_norm_eps", [1e-12, 1e-9, 1e-6]
    )
    new_config.initializer_range = trial.suggest_float(
        "initializer_range", 0.005, 0.05
    )

    new_config.loss_type = trial.suggest_categorical("loss_type", ["CE", "BPR"])

    return new_config


def suggest_lightgcn_cfg(config: DictConfig, trial: Trial) -> DictConfig:
    new_config = deepcopy(config)

    new_config.embedding_size = trial.suggest_categorical(
        "embedding_size", [16, 32, 64, 128]
    )
    new_config.n_layers = trial.suggest_int("n_layers", 1, 4)
    new_config.reg_weight = trial.suggest_categorical(
        "reg_weight", [1e-5, 1e-4, 1e-3, 1e-2]
    )

    return new_config


def suggest_gsau_cfg(config: DictConfig, trial: Trial) -> DictConfig:
    new_config = suggest_gsau_seq_cfg(config, trial)

    new_config.gamma = trial.suggest_float("gamma", 0.05, 0.5, step=0.1)

    return new_config




# def suggest_gsau_cfg(config: DictConfig, trial: Trial) -> DictConfig:
#     new_config = deepcopy(config)

#     new_config = suggest_gsau_seq_cfg(config, trial)

#     # new_config.n_layers = trial.suggest_int("n_layers", 1, 4)
#     # new_config.n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
#     # new_config.hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
#     # new_config.inner_size = trial.suggest_categorical("inner_size", [128, 256, 512])

#     new_config.hidden_dropout_prob = trial.suggest_float(
#         "hidden_dropout_prob", 0.0, 0.7, step=0.1
#     )
#     new_config.attn_dropout_prob = trial.suggest_float(
#         "attn_dropout_prob", 0.0, 0.7, step=0.1
#     )

#     new_config.hidden_act = trial.suggest_categorical("hidden_act", ["gelu", "relu", "swish"])
#     new_config.layer_norm_eps = trial.suggest_categorical("layer_norm_eps", [1e-12, 1e-9, 1e-6])
#     new_config.initializer_range = trial.suggest_float("initializer_range", 0.005, 0.05)
#     new_config.loss_type = trial.suggest_categorical("loss_type", ["CE", "BPR"])

#     return new_config


def run_gsau_recbole(config: DictConfig) -> dict:
    config_dict = OmegaConf.to_container(config, resolve=True)

    result = run_recbole(
        model="GSAU",
        dataset=config_dict["dataset"],
        config_file_list=config_dict.get("config_file_list"),
        config_dict=config_dict,
        saved=True,
        queue=None,
    )
    return result


class Objective:
    def __init__(self, base_cfg: DictConfig, out_dir: Path) -> None:
        self._base_cfg = base_cfg
        self._out_dir = out_dir

    def __call__(self, trial: Trial) -> float:
        suggested = suggest_gsau_cfg(self._base_cfg, trial)

        with open_dict(suggested):
            suggested["seed"] = int(suggested.get("seed", 42)) + int(trial.number)

        result = run_gsau_recbole(suggested)

        trial.set_user_attr("best_valid_score", float(result["best_valid_score"]))
        trial.set_user_attr("best_valid_result", dict(result["best_valid_result"]))
        trial.set_user_attr("test_result", dict(result["test_result"]))

        best_valid = result["best_valid_result"]
        score = best_valid.get(TARGET_METRIC)
        if score is None:
            raise KeyError(f"{TARGET_METRIC} not found in best_valid_result keys: {list(best_valid.keys())}")

        return float(score)


def run_optuna(
    base_cfg: DictConfig,
    experiment_name: str,
    num_trials: int,
    storage_dir: str = "./optuna_runs",
) -> None:
    out_dir = Path(storage_dir) / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(),
        study_name=experiment_name,
        storage=JournalStorage(JournalFileBackend(file_path=str(out_dir / f"{experiment_name}.log"))),
        load_if_exists=True,
    )

    study.optimize(Objective(base_cfg=base_cfg, out_dir=out_dir), n_trials=num_trials)

@click.command()
@click.option("--config_name", "-cn", type=str, required=True)
@click.option("--experiment_name", "-en", type=str, required=True)
@click.option("--num_trials", "-nt", type=int, required=True)
@click.option("--parallel_mode", "-pm", is_flag=True, default=False)
@click.option("--verbose", "-v", is_flag=True, default=True)
def main(
    config_name: str,
    experiment_name: str,
    num_trials: int,
    parallel_mode: bool,
    verbose: bool,
):
    out_dir = Path(OPTUNA_DIR) / experiment_name
    out_dir.mkdir(exist_ok=parallel_mode, parents=True)

    with initialize(config_path=CONFIG_DIR):
        base_cfg = compose(config_name=config_name)

    OmegaConf.set_struct(base_cfg, False)

    if verbose:
        print(OmegaConf.to_yaml(base_cfg))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(n_startup_trials=100),
        study_name=experiment_name,
        storage=JournalStorage(
            JournalFileBackend(file_path=str(out_dir / f"{experiment_name}.log"))
        ),
        load_if_exists=True,
    )

    study.optimize(
        Objective(base_cfg=base_cfg, out_dir=out_dir),
        n_trials=num_trials,
    )


if __name__ == "__main__":
    main()