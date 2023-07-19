@echo off
SETLOCAL
echo script name: %~nx0
SET /A args_count=0

FOR %%x IN (%*) DO (
    SET /A args_count+=1
)

IF NOT %args_count%==3 (
    echo Input illegal number of parameters %args_count%
    echo Need 3 parameters for dataset, tracking_status, and seed
    EXIT /B 1
)

IF "%TORCH_HOME%"=="" (
    echo Must set TORCH_HOME environment variable for data dir saving
    EXIT /B 1
) ELSE (
    echo TORCH_HOME : %TORCH_HOME%
)

SET dataset=cifar10
SET BN = 1
SET seed = 0
SET channel=16
SET num_cells=5
SET max_nodes=4
SET space=nas-bench-201

SET data_path="%TORCH_HOME%/cifar.python"

SET benchmark_file=%TORCH_HOME%\NAS-Bench-201-v1_1-096897.pth
SET save_dir=./output/search-cell-%space%/DARTS-V1-%dataset%-BN%BN%

SET OMP_NUM_THREADS=4 
python .\DARTS-V1.py ^
	--save_dir %save_dir% --max_nodes %max_nodes% --channel %channel% --num_cells %num_cells% ^
	--dataset %dataset% --data_path %data_path% ^
	--search_space_name %space% ^
	--config_path .\DARTS.config ^
	--arch_nas_dataset %benchmark_file% ^
	--track_running_stats %BN% ^
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 ^
	--workers 1 --print_freq 200 --rand_seed %seed%

ENDLOCAL