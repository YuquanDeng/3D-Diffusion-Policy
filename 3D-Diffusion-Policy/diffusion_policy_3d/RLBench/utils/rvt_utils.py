import argparse


def get_eval_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=["insert_onto_square_peg"])
    parser.add_argument(
        '--cameras',
        type=str,
        nargs='+',
        default=["front", "left_shoulder", "right_shoulder", "wrist"])
    parser.add_argument(
        '--model-folder',
        type=str,
        default=None)
    parser.add_argument(
        '--eval-datafolder',
        type=str,
        default='./data/val/')
    parser.add_argument(
        '--start-episode',
        type=int,
        default=0,
        help='start to evaluate from which episode')
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=10,
        help='how many episodes to be evaluated for each task')
    parser.add_argument(
        '--episode-length',
        type=int,
        default=25,
        help='maximum control steps allowed for each episode')
    parser.add_argument(
        '--headless',
        action='store_true',
        default=False)
    parser.add_argument(
        '--ground-truth',
        action='store_true',
        default=False)
    parser.add_argument(
        '--peract_official',
        action='store_true')
    parser.add_argument(
        '--mohits_model',
        action='store_true')
    parser.add_argument(
        '--peract_model_dir',
        type=str,
        default='runs/peract_official/seed0/weights/600000')
    parser.add_argument(
        '--device',
        type=int,
        default=0)
    parser.add_argument(
        '--record-every-n',
        type=int,
        default=5)
    parser.add_argument(
        '--log-name',
        type=str,
        default=None)
    parser.add_argument(
        '--ngc',
        action='store_true')
    parser.add_argument(
        '--mode',
        type=str,
        default='final',
        choices=['final', 'all', 'single'])
    parser.add_argument(
        '--model-name',
        type=str,
        default=None)
    parser.add_argument(
        '--use-input-place-with-mean',
        action='store_true')
    parser.add_argument(
        '--save-video',
        action='store_true')
    parser.add_argument(
        '--skip',
        action='store_true')
    parser.add_argument(
        '--ortho_cam',
        help='use orthographic input camera system',
        action='store_true')

    # Variations
    parser.add_argument(
        '--variations',
        type=int,
        nargs='+',
        default=[-1])
    
    # TODO: add excluded_variations args

    # VLM
    parser.add_argument(
        '--use_mask',
        action='store_true',
        help='use mask generated from VLM in evaluation',
        default=False)
    parser.add_argument(
        '--qwen_path',
        type=str,
        default=None)
    parser.add_argument(
        '--save_mask',
        action='store_true',
        help='save mask generated from rollout under log_dir',
        default=False)
    parser.add_argument(
        '--same_lang_goal',
        action='store_true',
        help='use same lang tokens for same RLbench tasks',
        default=False
    )
    parser.add_argument(
        '--mask_type',
        type=str,
        default="qwen")
    parser.add_argument(
        '--traj-datafolder',
        type=str,
        default=None)
    return parser