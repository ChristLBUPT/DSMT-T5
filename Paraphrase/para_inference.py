from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LogitsProcessor, LogitsProcessorList
from Paraphrase.para_datasets import ParaNMTDataset
from torch.utils.data import DataLoader
import torch
from torch import tensor
from tqdm import tqdm
import os
from post_process import main as post_process_main
from typing import Literal
import optuna

class TreeLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: T5Tokenizer) -> None:
        self.tok = tokenizer
        self.tgt_id = tokenizer('<tgt>', add_special_tokens=False).input_ids[0]
        self.square_b_ids = tokenizer('[[]', add_special_tokens=False).input_ids
        self.dot_id = tokenizer.convert_tokens_to_ids(').')
        self.comma_id = tokenizer.convert_tokens_to_ids('),')
        self.quote_id = tokenizer.convert_tokens_to_ids(');')
        self.rb_id = tokenizer.convert_tokens_to_ids(')')
        self.punctuation_ids = tokenizer('?!', add_special_tokens=False).input_ids[1:]
        self.lb_ids = tokenizer('((', add_special_tokens=False).input_ids
        self.eos_id = self.tok.eos_token_id
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for input_id, score in zip(input_ids, scores):
            # score[self.dot_id] = -float("inf")
            # score[self.comma_id] = -float("inf")
            # score[self.quote_id] = -float("inf")
            if (input_id == self.tgt_id).any():
                score[self.tgt_id] = -float("inf")
            for each in self.square_b_ids:
                score[each] = -float("inf")
            # if input_id[-1] == self.rb_id:
            #     for punc_id in self.punctuation_ids:
            #         score[punc_id] = -float("inf")

            # bm = 0
            # for each in input_id:
            #     if each in self.lb_ids:
            #         bm += 1
            #     if each == self.rb_id:
            #         bm -= 1

            # if torch.max(score, dim=-1).indices == self.eos_id:
            #     if bm > 0:
            #         score[self.rb_id] = 114514
            # if bm == 0 and (input_id == self.rb_id).long().sum() > 3:
            #     score[self.eos_id] = 114514
            # if torch.max(score, dim=-1).indices == self.dot_id:
            #     token2score = token2score = [*enumerate(score.tolist())]
            #     top10 = sorted(token2score, key=lambda x: x[1], reverse=True)[:10]
            #     print([self.tok.convert_ids_to_tokens([each]) for _, each in top10])
                # breakpoint()
            
        return scores

def inference(run_dir: str, use_checkpoint: int, dataset: str, data_mode: Literal['target', 'exemplar'], instance_format_name: str, **generation_kwargs):
    ckpt_dir = os.path.join(run_dir, 'checkpoints', f'{use_checkpoint}')
    tok: T5Tokenizer = T5Tokenizer.from_pretrained(ckpt_dir)
    model = T5ForConditionalGeneration.from_pretrained(ckpt_dir).to('cuda:0')
    # tok.add_special_tokens({"additional_special_tokens": ["<tgt>"]}, replace_additional_special_tokens=False)


    d = ParaNMTDataset(
        data_dir=f'./data/{dataset}/test',
        model='t5',
        tokenizer=tok,
        prune_height=5,
        has_ref=True,
        use_num_postfix=False,
        tgt_only=False,
        use_tgt_as_ref=False,
        instance_format_name=instance_format_name,
        max_length=1000
    )
    dl = DataLoader(d, 12, collate_fn=d.collate_fn)



    print(tok.decode(d[10][0]), tok.decode(d[10][1]))
    gen_kwargs = {
        "min_length": 5,
        "max_length": 1000,
        "temperature": 1.0,
        "do_sample": False,
        "top_k": 0,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,# if 'triplet' not in opt.instance_format_name else 2.0,
        "num_beams": 5,
        "no_repeat_ngram_size": 3
        # "bad_words_ids": [[628], [198]]
    }
    gen_kwargs.update(generation_kwargs)
    tree_lp = TreeLogitsProcessor(tok)
    results = [] 
    run_name = run_dir.split('/')[-1]
    os.makedirs(f'./runs/inference/{run_name}/results/{data_mode}/', exist_ok=True)
    with open(f'./runs/inference/{run_name}/results/{data_mode}/{use_checkpoint}_chunk0_raw.txt', 'w') as f:
        for idx, inputs in enumerate(tqdm(dl)):
            generated_sequences = model.generate(**inputs.to('cuda:0'), **gen_kwargs, logits_processor=LogitsProcessorList([tree_lp]) if 'triplet' in instance_format_name else None)
            decoded_sentences = tok.batch_decode(generated_sequences)
            f.write('\n'.join([each.replace('<pad>', '') for each in decoded_sentences]) + '\n')
            f.flush()
            results.extend(decoded_sentences)
            # for idx, single_sentence in enumerate(decoded_sentences):
            #     if single_sentence.find(').') > 0:
            #     # if single_sentence.find(']') > 0:
            #         print(tok.convert_ids_to_tokens(generated_sequences[idx]))
            #         breakpoint()
                    # raise StopIteration
    
    return post_process_main(f'inference/{run_name}', data_mode, './data/' + dataset + '/test', d.instance_format_name2extract_method[instance_format_name], use_checkpoint, )

def objective(trial: optuna.trial.Trial):
    num_beams = trial.suggest_categorical('num_beams', [1, 2, 3, 4, 5])
    length_penalty = trial.suggest_float('length_penalty', -0.5, 2.5, step=0.1)
    temperature = trial.suggest_float('temperature', 0.0, 1.0, step=0.1)
    pass
    

if __name__ == '__main__':
    inference('./runs/para_train_t5/_stage2/triplet_sentencetree_full_inetune_3/', 5, 'ParaNMT50m_triple', 'exemplar', "triplet_sentencetree")
    inference('./runs/para_train_t5/_stage2/triplet_sentencetree_fromscratch_3/', 5, 'ParaNMT50m_triple', 'exemplar', "triplet_sentencetree")