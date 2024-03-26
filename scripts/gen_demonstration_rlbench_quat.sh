# bash scripts/gen_demonstration_rlbench.sh light_bulb_in


# full rlbench task list:
# close_jar               light_bulb_in   open_drawer  place_shape_in_shape_sorter  push_buttons               put_item_in_drawer  reach_and_drag               stack_blocks  sweep_to_dustpan_of_size
# insert_onto_square_peg  meat_off_grill  place_cups   place_wine_at_rack_location  put_groceries_in_cupboard  put_money_in_safe   slide_block_to_color_target  stack_cups    turn_tap

cd RLBench

task_name=${1}


# if not use script policy, then use the expert policy trained with BAC.
use_script_policy=False

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_rlbench.py --env_name=${task_name} \
            --num_episodes 100 \
            --root_dir "/home/nil/manipulation/RVT2/rvt/data/rlbench_data/128x128/" \
            --save_dir "/home/nil/manipulation/3D-Diffusion-Policy/3D-Diffusion-Policy/data" \
            --use_point_crop True \
            --num_points 1024 \
            --use_script_policy $use_script_policy \
            --rot_representation quat
