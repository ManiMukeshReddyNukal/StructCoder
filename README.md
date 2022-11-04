
## Generating Dataset
After placing data at 
```./Data/ ```
Run below command
```
$ bash parse_data/preprocess_graphics.sh
```
Would identify codes which use Functions used in OpenGL to generate a set of dataset, on which we'll train our model on


## Running the module
Arguments to train the model
```
    parser.add_argument("--model_type", default="codet5", type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-base", type=str, 
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-base",
                        help="Pretrained tokenizer name or path if not the same as model_name")    
    parser.add_argument("--load_model_path", default="saved_models/pretrain/checkpoint-12000/pytorch_model.bin", type=str, 
                        help="Path to trained model: Should contain the .bin files" )  
    parser.add_argument("--config_name", default="Salesforce/codet5-base", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    
    ## Other parameters  
    parser.add_argument("--max_source_length", default=325, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_ast_depth", default=12, type=int)
    parser.add_argument("--max_target_length", default=155, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--alpha1", default=None, type=float)
    parser.add_argument("--alpha2", default=None, type=float)
    parser.add_argument("--alpha1_clip", default=-4, type=float)
    parser.add_argument("--alpha2_clip", default=-4, type=float)
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=500, type=int,
                        help="")
    parser.add_argument("--train_steps", default=100000, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
```

## StructCoder
Official implementation of [StructCoder: Structure-Aware Transformer for Code Generation](https://arxiv.org/abs/2206.05239)

## Setup the conda enviroment:
conda create -n structcoder --file structcoder.yml <br>
conda activate structcoder

## Download pretrained checkpoint:
mkdir -p saved_models/pretrain/checkpoint-12000 <br>
Download the pretrained model weights from [here](https://drive.google.com/drive/folders/1cyvtmZjaLc1OwlnU0_N_GwC_eAs5snf9?usp=sharing) and place it under saved_models/pretrain/checkpoint-12000/

## Finetune on translation:
python3 run_translation.py --do_train --do_eval --do_test --source_lang java --target_lang cs --max_target_length 320 --alpha1_clip -4 --alpha2_clip -4 

python3 run_translation.py --do_train --do_eval --do_test --source_lang cs --target_lang java --max_target_length 256 --alpha1_clip -4 --alpha2_clip -4

## Finetune on text-to-code generation:
python3 run_generation.py --do_train --do_eval --do_test
