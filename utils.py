import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_output(file: str, outputs: List[str]) -> None:
    outputs = list(map(lambda x: x + '\n', outputs))
    with open(file, 'w') as f:
        f.writelines(outputs)
        f.write('\n')


def load_input_data() -> List[str]:
    with open('test_mrs.txt', 'r+') as f:
        ls = f.readlines()
    return ls


def generate(model, tokenizer: AutoTokenizer, input: str) -> str:
    input = input.replace('\n','')
    txt = input + " <ref:> "
    inputs = tokenizer(txt, return_tensors='pt').to(device)

    o=model.generate(
        **inputs, 
        pad_token_id=tokenizer.eos_token_id, 
        num_beams=10,
        length_penalty=2, 
        max_length=512)
    output_text = tokenizer.decode(o[0], skip_special_tokens=True)
    output_text = output_text.replace(ls[i],'').lstrip()
    return output_text
