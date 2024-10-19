import argparse
from pathlib import Path

from egg.core.callbacks import WandbLogger

from game import ImageReconstructionGame, ImageDiscriminationGame, get_existing_models

##################################################################
# # To raise warnings
# import warnings
#
#
# def warning_to_error(message, category, filename, lineno, file=None, line=None):
#     raise category(message)
#
#
# warnings.filterwarnings("error", category=Warning)
# warnings.showwarning = warning_to_error  # Override the default warning handler

##################################################################


def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    def add_base_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--dataset', choices=['mnist', 'shapes'])
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--batch_limit', type=int, nargs=3,
                            help='number of batches to use for train, val and test. 0 means all batches')
        parser.add_argument('--length_cost', type=float, help='cost for message length in GS communication')
        parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
        parser.add_argument('--debug_mode', action=argparse.BooleanOptionalAction,
                            help='whether to use debug mode (small dataset, no saving, etc.)')
        parser.add_argument("--use_cuda", type=str2bool, nargs='?', const=True)
        parser.add_argument("--add_validation_callbacks", action=argparse.BooleanOptionalAction, help='add game-specific callbacks. logs to wandb if possible.')
        parser.add_argument('--log_every', type=int, help='how often to perform validation callbacks and log to wandb (epochs)')
        parser.add_argument('--sender_lr', type=float)
        parser.add_argument('--receiver_lr', type=float)
        parser.add_argument('--discretization_method',
                            choices=['gs', 'reinforce'],
                            help='optimization technique for discrete commnication.')
        parser.add_argument("--include_eos_token", type=str2bool, nargs='?', const=True, default=True, help='only relevant for GS-optimization. whether to include an EOS token which allows sentences shorter than max_len.')
        parser.add_argument('--architecture_type', choices=['conv_deconv', 'conv_pool'],
                            help='type of architecture for the vision models.')
        parser.add_argument('--vocab_size', type=int,
                            help='size of the vocabulary. irrelevant to quantize communication.')
        parser.add_argument('--legal_vocab_subsets', type=int, nargs='*',
                            help='sequence of subset sizes. If provided, each message will be composed of tokens'
                                 'from one of the subsets. Only relevant for generative Sender. Note that if'
                                 'include_eos_token=True, the first subset is one item smaller than its value.')
        parser.add_argument('--random_sender', action=argparse.BooleanOptionalAction, help='whether to use a random sender')
        parser.add_argument('--num_unique_random_messages', type=int,
                            help='number of unique messages used by the random sender. If not given, will use the entire channel.')
        parser.add_argument('--max_len', type=int,
                            help='maximum length of the message. if discretization method is quantize, this is the message dimension')
        parser.add_argument('--temperature', type=float, help='temperature for the GS communication')
        parser.add_argument('--use_temperature_scheduler', type=str2bool, nargs='?', const=True, default=True, help='temperature scheduler for GS')
        parser.add_argument('--temperature_decay', type=float, help='decay value for the temperature scheduler')
        parser.add_argument('--temperature_minimum', type=float, help='minimal temperature reachable by the scheduler')
        parser.add_argument('--temperature_update_freq', type=float, help='how often to apply the temperature scheduler (epochs)')
        # parser.add_argument('--cell', choices=['rnn', 'gru', 'lstm'], help='type of RNN cell')
        parser.add_argument('--trainable_temperature', action=argparse.BooleanOptionalAction)

        parser.add_argument('--encoder_pretraining_type',
                            choices=['autoencoder', 'supervised', 'contrastive', 'none'],
                            help='type of continuous pretraining for the encoder. If not provided or \'none\', no pretraining')
        parser.add_argument('--callback_perceptual_model_pretraining_type',
                            choices=['autoencoder', 'supervised', 'contrastive', 'none'],
                            help='type of continuous pretraining for the perceptual model used for callbacks.')
        parser.add_argument('--encoder_cpt_name', type=str, help='name of the checkpoint for the encoder')
        parser.add_argument('--callback_perceptual_model_cpt_name', type=str, help='name of the checkpoint for perceptual model used for callbacks. If not provided, no perceptual encoding.')
        parser.add_argument('--callback_perceptual_model_architecture_type', choices=['conv_deconv', 'conv_pool'],
                            help='type of architecture for perceptual model used for callbacks.')
        parser.add_argument('--use_game_encoder_as_perceptual_model', action=argparse.BooleanOptionalAction,
                            help='whether to use the game encoder as the perceptual model used for callbacks (overrides the perceptual_model arguments)')
        parser.add_argument("--frozen_encoder", type=str2bool, nargs='?', const=True,
                            help='whether to freeze the encoder (and only train the RNN part of Sender)')
        parser.add_argument("--use_cached_encoder", type=str2bool, nargs='?', const=True,
                            help='whether to use pre-calculated image embeddings. Only relevant if frozen_encoder=True.'
                                 'Requires that the encodings are already cached, via the caching method in scripts.py')
        parser.add_argument('--sender_gen_model',
                            choices=['rnn', 'gru', 'lstm'],
                            help='type of message generator')
        parser.add_argument('--receiver_embed_model', choices=['rnn', 'gru', 'lstm'],
                            help='type of message embedder')
        parser.add_argument('--sender_RNN_hidden_dim', type=int)
        parser.add_argument('--receiver_RNN_hidden_dim', type=int)
        parser.add_argument('--sender_RNN_emb_size', type=int)
        parser.add_argument('--receiver_RNN_emb_size', type=int)
        parser.add_argument('--receiver_RNN_num_layers', type=int)

    def add_train_arguments(parser: argparse.ArgumentParser):
        # the arguments that aren't part of any game's config
        parser.add_argument("--all_dimensions", type=int, default=None,
                            help='value to set all dimensions to, for quick tests')
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument("--use_wandb_logger", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--save_checkpoint", type=str2bool, nargs='?', const=True, default=True,
                            help='whether to save checkpoints')
        parser.add_argument('--save_name', type=str, default=None,
                            help='name of the saved model. Defaults to {datset}_model')
        parser.add_argument('--load_name', choices=get_existing_models(), default=None,
                            help='name of a model to load. If not provided, no model is loaded. If provided, all other arguments are ignored excpet epochs, save_name and use_wandb_logger')

    argparser = argparse.ArgumentParser(description='process hyper-parameters')
    subparsers = argparser.add_subparsers(required=True, help='which function to run')

    parser_reco = subparsers.add_parser('reconstruction', help='Train a Reconstruction game')
    default_reco = ImageReconstructionGame.Config(
        # dataset='mnist',
        # encoder_pretraining_type='autoencoder',
        # callback_perceptual_model_pretraining_type='autoencoder',
        # decoder_pretraining_type='autoencoder',
        # encoder_cpt_name='pretrained_enc1',
        # decoder_cpt_name='pretrained_enc1',
        batch_limit=(0, 0, 1)
    )
    add_base_arguments(parser_reco)
    add_train_arguments(parser_reco)

    parser_reco.add_argument('--receiver_type', choices=['deconv', 'MLP'], help='type of Receiver agent')
    parser_reco.add_argument('--decoder_pretraining_type', choices=['autoencoder', 'none'],
                             help='type of continuous pretraining for the decoder. If not provided or \'none\', no pretraining')
    parser_reco.add_argument('--decoder_cpt_name', type=str, help='name of the checkpoint for the decoder')
    parser_reco.add_argument("--frozen_decoder", type=str2bool, nargs='?', const=True,
                             help='whether to freeze the decoder (and only train the RNN part of Receiver)')
    parser_reco.add_argument('--receiver_MLP_num_layers', type=int, dest='MLP_num_hidden_layers')
    parser_reco.add_argument('--receiver_MLP_activation', choices=['relu', 'lrelu', 'sigmoid', 'tanh'],
                             dest='MLP_activation')
    parser_reco.add_argument('--receiver_MLP_hidden_dim', type=int, dest='MLP_hidden_dim')
    parser_reco.add_argument('--loss_mse_weight', type=float, help='weight of the MSE loss')
    parser_reco.add_argument('--loss_perceptual_weight', type=float,
                             help='weight of the perceptual loss. checkpoint must be provided if != 0')
    parser_reco.add_argument('--loss_perceptual_checkpoint', type=str, help='checkpoint for the perceptual loss')
    parser_reco.add_argument('--loss_contrastive_weight', type=float,
                             help='weight of the contrastive loss. checkpoint must be provided if != 0')
    parser_reco.add_argument('--loss_contrastive_checkpoint', type=str, help='checkpoint for the contrastive loss')
    parser_reco.add_argument('--loss_autoencoder_weight', type=float,
                             help='weight of the continuous autoencoder loss. checkpoint must be provided if != 0')
    parser_reco.add_argument('--loss_autoencoder_checkpoint', type=str, help='checkpoint for the continuous autoencoder loss. Giving the same checkpoint as the pretraiend encoder and decoder cpt results in distillation loss.')
    parser_reco.add_argument('--callback_num_distractors', type=int, help='number of distractors in discrimination accuracy callback')
    parser_reco.set_defaults(**default_reco.get_config_dict(), game_type='reconstruction')

    parser_disc = subparsers.add_parser('discrimination', help='Train a Discrimination game')
    default_disc = ImageDiscriminationGame.Config(
        # dataset='mnist',
        # encoder_pretraining_type='autoencoder',
        # callback_perceptual_model_pretraining_type='autoencoder',
        # receiver_encoder_pretraining_type='autoencoder',
        # encoder_cpt_name='pretrained_enc1',
        # receiver_encoder_cpt_name='pretrained_enc1',
        batch_limit=(0, 0, 1)
    )
    add_base_arguments(parser_disc)
    add_train_arguments(parser_disc)

    parser_disc.add_argument('--receiver_encoder_pretraining_type', choices=['autoencoder', 'supervised',
                                                                             'contrastive', 'none'],
                             help='type of continuous pretraining for receiver\'s encoder. If not provided or \'none\', no pretraining')
    parser_disc.add_argument('--receiver_encoder_cpt_name', type=str, help='name of the checkpoint for the decoder')
    parser_disc.add_argument("--receiver_frozen_encoder", type=str2bool, nargs='?', const=True,
                             help='whether to freeze Receiver\'s encoder (and only train the RNN and MLP parts of Receiver)')
    parser_disc.add_argument('--receiver_MLP_num_layers', type=int, dest='MLP_num_hidden_layers')
    parser_disc.add_argument('--receiver_MLP_activation', choices=['relu', 'lrelu', 'sigmoid', 'tanh'],
                             dest='MLP_activation')
    parser_disc.add_argument('--receiver_MLP_hidden_dim', type=int, dest='MLP_hidden_dim')
    parser_disc.add_argument('--num_distractors', type=int, help='number of distractors in training. Currently must be strictly smaller than batch_size')
    parser_disc.add_argument('--discrimination_strategy',
                             choices=['vanilla', 'supervised', 'contrastive', 'classification'],
                             help='how to choose distractors and calculate loss (for both training and evaluation).')
    parser_disc.set_defaults(**default_disc.get_config_dict(), game_type='discrimination')

    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    if args.debug_mode:
        # low dimensions, epochs, and no saving\logging to wandb.
        args.epochs = 2
        args.batch_limit = (3, 3, 3)
        args.save_checkpoint = False
        args.use_wandb_logger = False
        args.all_dimensions = 10
    for k,v in vars(args).items():
        if isinstance(v, str) and v.lower() == 'none':
            setattr(args, k, None)
    if args.all_dimensions is not None:
        # set all hidden dims and embed sizes of the communication channel to the given value
        val = args.all_dimensions
        for attr in args.__dict__:
            if any(substring in attr for substring in ['hidden_dim', 'emb_size']):
                setattr(args, attr, val)
    if not args.save_checkpoint:
        print('Not saving checkpoints!')
    game_class = {'reconstruction': ImageReconstructionGame,
                  'discrimination': ImageDiscriminationGame}[args.game_type]
    if args.load_name is not None:
        game = game_class.load_from_checkpoint(args.load_name)
    else:
        config_args = {k: v for k, v in vars(args).items() if k in game_class.Config.__slots__}
        missing_args = set(game_class.Config.__slots__) - set(vars(args).keys())
        if args.verbose and missing_args:
            print('These config args are not parsed:\n', missing_args)
        config = game_class.Config(**config_args)
        game = game_class(config)

    game.train(epochs=args.epochs,
               save_checkpoint=args.save_checkpoint,
               use_wandb_logger=args.use_wandb_logger,
               save_name=args.save_name)

    loss, _ = game.eval(subset='train')
    print("train loss:", loss)
    loss, _ = game.eval(subset='val')
    print("validation loss:", loss)

    if args.use_wandb_logger:
        callbacks = game.trainer.callbacks
        for callback in callbacks:
            if isinstance(callback, WandbLogger):
                callback.log_to_wandb({"final_val_loss": loss}, commit=True)
                break


if __name__ == '__main__':
    main()
