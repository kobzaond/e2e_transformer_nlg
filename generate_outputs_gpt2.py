from datasets import ClassLabel, load_dataset, Value 
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Specify output path.')
    parser.add_argument("--output_dir", type=str, required=True, help="Directory of the model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.output_dir+'/'+args.checkpoint).to(device)
    f = open('test_mrs.txt', 'r+')
    ls = f.readlines()
    f.close()

    outputs= []
    for i in range(len(ls)):
        ls[i] = ls[i].replace('\n','')
        txt = ls[i] + " <ref:> "
        inputs = tokenizer(txt, return_tensors='pt').to(device)

        o=model.generate(
            **inputs, 
            pad_token_id=tokenizer.eos_token_id, 
            num_beams=10,
            length_penalty=2, 
            max_length=512)
        output_text = tokenizer.decode(o[0], skip_special_tokens=True)
        output_text = output_text.replace(ls[i],'').lstrip()
        outputs.append(output_text+'\n')

    f = open('outputs.txt','w')
    f.writelines(outputs)
    f.close()