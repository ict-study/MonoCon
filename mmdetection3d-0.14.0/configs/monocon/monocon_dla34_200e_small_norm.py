_base_ = [
    '../_base_/models/monocon_dla34_norm.py',
    '../_base_/datasets/small-mono3d-3class-monocon.py',
    '../_base_/schedules/cyclic_40e.py',
    '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=5)

workflow = [('train', 1)]
mp_start_method = 'fork'
find_unused_parameters = True