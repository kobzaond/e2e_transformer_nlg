# e2e_transformer_nlg
solving the e2e challenge by two transformer generative models: T5-base and gpt2

## Training model
This repo contains two scripts - one to train the GPT2 and one to train the T5 model.
According to the automatic metrics considered in the E2E challenge

## Generating outputs:
This repo contains two scripts for output generation, each for T5 or GPT2.
Both scripts use beam search

## Usage:
```
pip3 install -r requirements.txt
```

```
python3 train_gpt2.py --output_dir $OUTPUT_DIR

or

python3 train_t5.py --output_dir $OUTPUT_DIR
```

```
python3 make_test_refs.py
```


```
python3 generate_outputs_t5.py --model_path $OUTPUT_DIR --checkpoint $CHECKPOINT

or 

python3 generate_outputs_gpt2.py --model_path $OUTPUT_DIR --checkpoint $CHECKPOINT
```

the system will generate a file `outputs.txt` in the current directory.

## Results:
* results are obtained on the e2e test set
### GPT2:
SCORES:

BLEU: 0.6768
NIST: 8.6796
METEOR: 0.4595
ROUGE_L: 0.6818
CIDEr: 2.3878

### T5-base:
SCORES:

BLEU: 0.6421
NIST: 8.1828
METEOR: 0.4576
ROUGE_L: 0.6901
CIDEr: 2.3272

## Evaluation:
For automatic metric evaluation, clone the official repo: https://github.com/tuetschek/e2e-metrics
and use the outputs.txt and test_references.txt as source and targets.
