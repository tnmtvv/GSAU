import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

log_path = "optuna_outputs/gsau_tune_02/gsau_tune_02.log"
study_name = "gsau_tune_02"

storage = JournalStorage(JournalFileBackend(file_path=log_path))
study = optuna.load_study(study_name=study_name, storage=storage)

print("best_value:", study.best_value)
print("best_trial:", study.best_trial.number)
print("best_params:", study.best_trial.params)
print("best_user_attrs:", study.best_trial.user_attrs)
