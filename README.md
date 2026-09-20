# GA-GRU + GAT Rumor Detection

Compact implementation of the final GA-GRU + GAT rumor-detection pipeline for
PHEME, Weibo, and DRWeibo.

## Configuration check

```powershell
$python = 'D:\conda_envs\rumor_detection_pheme_ft\python.exe'
$root = 'D:\project\project\R-gat+bigru\dual_branch_rumor_graph_only_copy'
$env:PYTHONPATH = $root
& $python "$root\scripts\run_gagru_gat.py" pheme --mode check
& $python "$root\scripts\run_gagru_gat.py" weibo --mode check
& $python "$root\scripts\run_gagru_gat.py" drweibo --mode check
```
