

gcc -o cache_sets_locator cache_sets_locator.c -ldl --> compile prime and probe (*)
sudo ./cache_sets_locator / sudo taskset -c 0 ./cache_sets_locator --> run prime and probe (*)
python3 victim_model.py / taskset -c 0 python3 victim_model.py --> run victim AI model (*)
python3 find_full_inference.py --> find which inference are fully inside spike_log.txt (*)

python3 plot_spikes_vs_layers.py --> plot itcopy, oncopy, kernel latency
python3 con_arch.py --> print number of spikes
python3 cluster.py --> cluster spikes based on time window of specific one inference (*)
python3 pattern.py --> pattern to find layer type (conv, fc). (*)
python3 layer.py --> input/output dimension based on tiles 


(*) -> used in final experiment
