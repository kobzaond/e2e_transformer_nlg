from datasets import ClassLabel, load_dataset, Value 
import torch
from utils import save_output


if __name__=='__main__':
    """
    Creates references for evaluation and stores MRs into a file (input for a trained generative model)
    """

    data = load_dataset('e2e_nlg')['test']
    dictic: Dict[str, List[str]] = {}
    dataloader = torch.utils.data.DataLoader(data, batch_size=1)
    ls=[]

    for item in dataloader:
        key = item['meaning_representation'][0]
        value = item['human_reference'][0]
        if key not in ls:
            ls.append(key)
        if key not in dictic:
            dictic[key] = [value]
        else:
            dictic[key].append(value)

    # save files:
    save_output('test_mrs.txt', ls)

    with open('test_references.txt','w+') as f:
        for i, key in enumerate(ls):
            f.writelines(list(map(lambda x: x + '\n', dictic[key])))
            if i<len(ls)-1:
                f.write('\n')