from datasets import ClassLabel, load_dataset, Value 
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, IntervalStrategy 
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
import torch.nn.functional as F
import torch


def tokenizer_helper(tokenizer: AutoTokenizer, mrs: List[str], references: List[str]) -> Dict[str, torch.Tensor]:
    dictic = {}
    inputs = tokenizer(list(
                map(
                    lambda a: a + " <ref>: " , mrs)), 
                    truncation="only_second",max_length=128, padding='max_length')
    labels = tokenizer(references, max_length=512-128, truncation='only_second', padding='max_length')
    for key in inputs.keys():
        dictic[key] = inputs[key]
    dictic['labels'] = labels['input_ids']
    return dictic


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Specify output path.')
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where to save the model.")
    args = parser.parse_args()

    # load data
    data = load_dataset('e2e_nlg')

    # add special token: separator between MR and a corresponding human ref.
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('reference_tag')
    tokenizer.add_special_tokens({'reference_tag':"<ref>:"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer._reference_tag = '<ref>:'

    # tokenize data
    data=data.map(
        lambda x : tokenizer_helper(tokenizer, x['meaning_representation'], x['human_reference']), batched=True)

    # load T5-base model, resize embeddings
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,#"/mnt/proj1/open-24-17/e2e_t5",
        num_train_epochs=6,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=16,
        learning_rate=9e-5,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        weight_decay=1e-2,
        lr_scheduler_type='linear',
        load_best_model_at_end = True,
        run_name='e2e',
        report_to=None
    )

    # fine-tune the model; the evaluation metric is the eval loss
    trainer = Trainer(
            model=model, args=training_args, train_dataset=data['train'], eval_dataset=data['validation'],
        )


    trainer.train()

    #save tokenizer; the model checkpoints are saved automatically by the huggingface trainer
    tokenizer.save_pretrained(args.output_dir)