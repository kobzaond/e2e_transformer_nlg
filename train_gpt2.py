from datasets import ClassLabel, load_dataset, Value 
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, TrainingArguments, Trainer, IntervalStrategy 
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Specify output path.')
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where to save the model.")
    args = parser.parse_args()

    # load data
    reporting_enabled = True
    data = load_dataset('e2e_nlg')

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # add special token: separator between MR and a corresponding human ref.
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('reference_tag')
    tokenizer.add_special_tokens({'reference_tag':"<ref:>"})
    tokenizer._reference_tag = '<ref>:'
    tokenizer.pad_token = tokenizer.eos_token

    # tokenize data
    data=data.map(
        lambda x : tokenizer(
            list(
                map(
                    lambda a,b: a + " <ref:> " + b, x['meaning_representation'], x['human_reference'])), 
                    truncation="only_second",max_length=512, padding='max_length'), batched=True
                    )
    data = data.map(lambda x: {'labels':x['input_ids']})
    
    # load T5-base model, resize embeddings
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,#"/mnt/proj1/open-24-17/e2e_gpt2",
        num_train_epochs=6,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
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
    tokenizer.save_pretrained(args.output_dir)