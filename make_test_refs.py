from datasets import ClassLabel, load_dataset, Value 
import torch


if __name__=='__main__':
    """
    Creates references for evaluation and stores MRs into a file (input for a trained generative model)
    """

    data = load_dataset('e2e_nlg')['test']
    dictic = {}
    dataloader = torch.utils.data.DataLoader(data, batch_size=1)
    ls=[]

    for item in dataloader:
        key = item['meaning_representation'][0]
        value = item['human_reference'][0]
        if key+'\n' not in ls:
            ls.append(key+'\n')
        if key+'\n' not in dictic:
            dictic[key+'\n'] = [value+'\n']
        else:
            dictic[key+'\n'].append(value+'\n')

    # save files:
    f = open('test_mrs.txt', 'w+')
    f.writelines(ls)
    f.close()
    f = open('test_references.txt','w+')
    for i, key in enumerate(ls):
        f.writelines(dictic[key])
        if i<len(ls)-1:
            f.write('\n')
    f.close()