import argparse

from game_parts.continuous_training import train_vision_model, make_loss_module

"""
command example (slurm):
srun -p nlp -A nlp --gres=gpu:1 python pretrain.py --model_type=supervised --dataset=mnist --epochs=50 --save_name=try
"""


def parse_args():
    argparser = argparse.ArgumentParser(description='process parameters for continuous pre-training')
    argparser.add_argument('--model_type', choices=['autoencoder', 'supervised', 'contrastive'], default='autoencoder')
    argparser.add_argument('--architecture_type', choices=['conv_deconv', 'conv_pool'], default='conv_deconv')
    argparser.add_argument('--dataset', choices=['mnist', 'shapes'], default='mnist')
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--save_name', type=str, default=None,
                           help='name of the saved model. Defaults to continuous_model')
    argparser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--start_checkpoint_name', type=str, default=None,
                           help='Optionally, checkpoint from which to start training. only save name, not full file name.')
    argparser.add_argument('--debug_mode', action=argparse.BooleanOptionalAction, help='overfit on a single batch')
    argparser.add_argument('--use_wandb_logger', action=argparse.BooleanOptionalAction, help='only relevant for autoencoder, with debug mode = False')
    argparser.add_argument('--batch_limit', type=int, nargs=3, help='number of batches to use for train, val and test. 0 means all batches', default=(0, 0, 0))
    argparser.add_argument('--num_augmentations', type=int, default=2, help='number of augmentation of each image, for contrastive training')
    argparser.add_argument('--use_labels', action=argparse.BooleanOptionalAction, help='whether to use SupCon rather than SimCLS (add supervised labels)')
    argparser.add_argument('--projection_dim', type=int, default=50)
    argparser.add_argument('--loss_mse_weight', type=float, default=1.0, help='weight of the MSE loss')
    argparser.add_argument('--loss_perceptual_weight', type=float, default=0.0,
                           help='weight of the perceptual loss. checkpoint must be provided if > 0')
    argparser.add_argument('--loss_perceptual_checkpoint', type=str, default='NOT_PROVIDED',
                           help='checkpoint for the perceptual loss: {encoder_pretraining_type}_{architecture_type}_{dataset}_{cpt_name}')
    argparser.add_argument('--loss_contrastive_weight', type=float, default=0.0,
                           help='weight of the contrastive loss. checkpoint must be provided if > 0')
    argparser.add_argument('--loss_contrastive_checkpoint', type=str, default='NOT_PROVIDED',
                           help='checkpoint for the contrastive loss: {encoder_pretraining_type}_{architecture_type}_{dataset}_{cpt_name}')
    argparser.add_argument('--loss_autoencoder_weight', type=float, default=0.0,
                           help='weight of the continuous autoencoder loss. checkpoint must be provided if != 0')
    argparser.add_argument('--loss_autoencoder_checkpoint', type=str,
                           help='checkpoint for the continuous autoencoder loss. Giving the same checkpoint as the pretraiend encoder and decoder cpt results in distillation loss.')

    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    if args.save_name is None:
        args.save_name = "continuous_model"
    if args.model_type == 'autoencoder':
        args.autoencoding_criterion = make_loss_module(args.loss_mse_weight,
                                                       args.loss_perceptual_weight,
                                                       args.loss_perceptual_checkpoint,
                                                       args.loss_contrastive_weight,
                                                       args.loss_contrastive_checkpoint,
                                                       args.loss_autoencoder_weight,
                                                       args.loss_autoencoder_checkpoint)
    train_vision_model(**vars(args))


if __name__ == '__main__':
    main()
