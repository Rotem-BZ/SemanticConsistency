# Semantics and Spatiality of Emergent Communication
## Code dependencies:
* pytorch
* torchvision
* pytorch lightning
* wandb
* matplotlib
* datasets
* wget
* EGG: https://github.com/facebookresearch/EGG
* shapeworld: https://github.com/AlexKuhnle/ShapeWorld

Note that the shapeworld package does not work on Windows.

### A conda environment with the required dependencies:
1. `git clone https://github.com/Rotem-BZ/SemanticConsistency.git`
2. `conda env create -f semantic_consistency_env.yaml`
3. `conda activate sc_env`

# Argparsing guide
To launch EC training, run the `train.py` script. The first argument is the game type,
with two options: `reconstruction` or `discrimination`. Each option has slightly
different arguments.
## Data, training and logging arguments
* `--dataset`: out of `shapes`,`mnist`
* `--batch_size`
* `--epochs`
* `--batch_limit`: how many batches from each subset (train, val, test) to use. 0 means all batches (default). Three arguments.
* `--sender_lr`
* `--receiver_lr`
* `--verbose`: boolean action. prints out the config, agents and uses a ProgressBarLogger.
* `--use_wandb_logger`: boolean (defaults to True, set to False/no/f/n/0 to change).
* `--save_checkpoint`: boolean (defaults to True, set to False/no/f/n/0 to change). Whether to save checkpoints.
* `--save_name`: name of the saved model. Defaults to \[dataset\]_model
* `--load_name`: Optional name of a start checkpoint. If provided, all other arguments are ignored excpet `epochs`, `save_name` and `use_wandb_logger`.
* `--debug_mode`: sets all the dimensions to 10, epochs to 2 with 3 batches each, disables checkpoint saving and wandb logging.
* `--use_cuda`
* `--add_validation_callbacks`: Adds evaluation metrics to be reported during training. See `callbacks.py` for the implemented callbacks.
* `--log_every`: how often (epochs) to perform callbacks.
## EC setup arguments
The implemented discrete communication methods are Gumbel-Softmax and Reinforce.
* `--discretization_method`: `gs` or `reinforce`
* `--vocab_size`
* `--max_len`
### Gumbel-Softmax
* `--length_cost`: cost for message length in GS communication.
* `--include_eos_token`: only relevant to `gs` communication, whether to use an EOS token.
* `--temperature`: temperature for the Gumbel-Softmax discretization method
* `--use_temperature_scheduler`: boolean (defaults to False, set to True/yes/t/y/1 to change).
* `--temperature_decay`
* `--temperature_minimum`
* `--temperature_update_freq`
* `--trainable_temperature`
## Vision encoder arguments
The image data is passed through a CNN as part of Sender, as well as for language analysis (metrics like TopSim may use
[perceptual distance](https://ieeexplore.ieee.org/abstract/document/9207431) rather than pixel distance). The Vision
encoder can be pretrained via `pretrain.py`, trained jointly with the rest of the EC setup, or both.
* `--encoder_pretraining_type`: Which type of pretrained Vision encoder to load for Sender. The types correspond to pretraining objectives; `autoencoder`, `supervised`, `contrastive` or `none`.
* `--encoder_cpt_name`: Checkpoint to load.
* `--architecture_type`: Architecture of the pretrained Vision encoder, either `conv_deconv` or `conv_pool`.
* `--frozen_encoder`: boolean (defaults to True, set to False/no/f/n/0 to change). Whether the pretrained encoder should be fixed during EC training.
* `--use_cached_encoder`: if the encoder is frozen, we can speed up training by using cached extracted representations rather than computing the image encodings on the fly. To use this feature, run `scripts.py` to cache the encoder outputs after training the continuous model.
* `--callback_perceptual_model_pretraining_type`: pretraining type for the perceptual model used for callbacks (`autoencoder`, `supervised`, `contrastive` or `none`.).
* `--callback_perceptual_model_cpt_name`:  Checkpoint to load.
* `--callback_perceptual_model_architecture_type`: Architecture of the perceptual model, either `conv_deconv` or `conv_pool`.
* `--use_game_encoder_as_perceptual_model`: Ignores the previous three arguments and uses Sender's encoder as perceptual for the callbacks. Boolean action.
## Message generation and embedding arguments
When the discretization method is `gs` or `reinforce`, a sequential model is used by Sender to generate the message and
another sequential model is used by Receiver to embed the message.
* `--sender_gen_model`: `rnn`, `gru`, `lstm`.
* `--receiver_embed_model`: `rnn`, `gru`, `lstm`.
* `--all_dimensions`: A quick way to set all hidden dimensions and embedding sizes to the given integer.
* `--sender_RNN_hidden_dim`
* `--receiver_RNN_hidden_dim`
* `--sender_RNN_emb_size`
* `--receiver_RNN_emb_size`
* `--receiver_RNN_num_layers`
## Special modes
* `--legal_vocab_subsets`: Any number of positive integers that add up to `vocab_size`. Relevant only to GS communication. When given, the vocabulary splits into subsets of the given sizes, and then messages may only contain symbols from a single subset, effectively creating clusters of legal messages.
* `--random_sender`: Relevant only to GS communication. replaces Sender with a random messages generator. This generator will be consistent, i.e., always output the same message for the same input. It also supports the legal_vocab_subsets argument.
* `--num_unique_random_messages`: Optionally, limit the number of unique messages generated by the random sender.
## Task-specific arguments
### Reconstruction
Receiver generates an image by either transpose-convolution or MLP.
* `--receiver_type`: type of Receiver agent: `deconv` or `MLP`
* `--decoder_pretraining_type`: if `deconv` Receiver, type of pretrained decoder to load, `autoencoder` or `none`.
* `--decoder_cpt_name`: name of decoder checkpoint.
* `--frozen_decoder`: whether decoder is frozen during EC training. boolean (defaults to True, set to False/no/f/n/0 to change).
* `--receiver_MLP_num_layers`
* `--receiver_MLP_hidden_dim`
* `--receiver_MLP_activation`: `relu`, `lrelu`, `sigmoid` or `tanh`
* `--loss_mse_weight`: multiplier for the regular pixel-wise MSE loss
* `--loss_perceptual_checkpoint`: optional checkpoint of a pretrained encoder to be used as perceptual metric for training.
* `--loss_perceptual_weight`: multiplier for the perceptual loss.
* `--loss_contrastive_checkpoint` optional checkpoint of a contrastive-learning pretrained encoder, which can generate distance score between the input and reconstruction, used as loss.
* `--loss_contrastive_weight`: multiplier for the contrastive loss.
* `--loss_autoencoder_checkpoint`: optional checkpoint of a pretrained autoencoder. The autoencoder loss is the distance between the game reconstruction and the pretrained model's reconstruction of the same input. 
* `--loss_autoencoder_weight`: multiplier for the autoencoder loss.
### Discrimination
Receiver maps candidate images through a Vision encoder module followed by an MLP. After normalization, the dot product
between image and message representations is optimized via the infoNCE loss.
* `--receiver_encoder_pretraining_type`: type of pretrained encoder for Receiver's Vision module
* `--receiver_encoder_cpt_name`: checkpoint for Receiver's encoder.
* `--receiver_frozen_encoder`: whether Receiver's encoder should be frozen during EC training. boolean (defaults to True, set to False/no/f/n/0 to change).
* `--receiver_MLP_num_layers`
* `--receiver_MLP_activation`: `relu`, `lrelu`, `sigmoid` or `tanh`
* `--receiver_MLP_hidden_dim`
* `--num_distractors`
* `--discrimination_strategy`: how to choose distractors. `vanilla`: random distractors. `supervised`: different-label distractors.

# Reproducibility
The commands we have used in our experiments are the following.
## 1.Vision pretraining
shapes:
```
python pretrain.py --model_type autoencoder --architecture_type conv_deconv --dataset shapes --save_name shapes_ae --use_wandb_logger
```
mnist:
```
python pretrain.py --model_type autoencoder --architecture_type conv_deconv --dataset mnist --save_name mnist_ae --use_wandb_logger
```
After this command, you can run `scripts.py` to cache the encoder outputs and then use `--use_cached_encoder` in the `train.py` call to speed up training.
## 2. EC training
Note 1: if you wish to train agents from scratch, you can skip the Vision pretraining step and replace every "x_pretraining_type" to "none", every "frozen_x" to False, and "--use_cached_encoder=False".

Note 2: For the cluster variance experiment, we used `--legal_vocab_subsets 2 2 2 2 2` in addition to any of the following calls.
### Reconstruction
```
python train.py reconstruction --dataset=shapes --encoder_cpt_name=shapes_ae --decoder_cpt_name=shapes_ae --all_dimensions=150 --add_validation_callback --use_game_encoder_as_perceptual_model --log_every=10 --epochs=200 --save_checkpoint=False --callback_num_distractors=40
```
```
python train.py reconstruction --dataset=mnist --encoder_cpt_name=mnist_ae --decoder_cpt_name=mnist_ae --all_dimensions=150 --add_validation_callbacks --use_game_encoder_as_perceptual_model --log_every=10 --epochs=100 --save_checkpoint=False --callback_num_distractors=40
```
### Discrimination
```
python train.py discrimination --dataset=shapes --encoder_cpt_name=shapes_ae --receiver_encoder_cpt_name=shapes_ae --receiver_MLP_num_layers=2 --num_distractors=40 --discrimination_strategy=vanilla --log_every=10 --all_dimensions=150 --epochs=200 --add_validation_callbacks --use_game_encoder_as_perceptual_model
```
```
python train.py discrimination --dataset=mnist --encoder_cpt_name=mnist_ae --receiver_encoder_cpt_name=mnist_ae --receiver_MLP_num_layers=2 --num_distractors=40 --discrimination_strategy=vanilla --log_every=10 --all_dimensions=150 --epochs=100 --add_validation_callbacks --use_game_encoder_as_perceptual_model
```
### Supervised discrimination
Only implemented on MNIST:
```
python train.py discrimination --dataset=mnist --encoder_cpt_name=mnist_ae --receiver_encoder_cpt_name=mnist_ae --receiver_MLP_num_layers=2 --num_distractors=40 --discrimination_strategy=supervised --log_every=10 --all_dimensions=150 --epochs=100 --add_validation_callbacks --use_game_encoder_as_perceptual_model
```
