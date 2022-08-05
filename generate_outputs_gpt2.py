from datasets import ClassLabel, load_dataset, Value 
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import argparse
from utils import load_input_data, save_output, generate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Specify output path.')
    parser.add_argument("--output_dir", type=str, required=True, help="Directory of the model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.output_dir+'/'+args.checkpoint).to(device)

    test_data = load_input_data()

    outputs = [generate(model, tokenizer, sample) for sample in test_data] 

    save_output('output.txt', outputs)
    