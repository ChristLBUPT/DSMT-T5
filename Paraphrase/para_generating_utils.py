from transformers import BartForConditionalGeneration, BartTokenizer, PreTrainedTokenizer, PreTrainedModel, Constraint
from typing import Union, Dict, List
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
import re
import logging
import os
from argparse import Namespace
from tqdm import tqdm, trange
from time import sleep
from data.TreeUtils.tree import MyTree as Tree  
from torch import distributed as dist
from multiprocessing import Process
import pickle as pkl


logger = logging.getLogger(__name__)# + f'r{dist.get_rank()}' if dist.is_available() else '')

# def extract_tgt_sentence_from_outputs()



def teacher_forcing_generate(model: BartForConditionalGeneration, tok: BartTokenizer, src_sent: Union[str, Tensor], debug=False, **generation_kwargs):
    right_bracket_id = tok(' )', add_special_tokens=False).input_ids
    bracket_ids = tok(') )', add_special_tokens=False).input_ids + tok('( (', add_special_tokens=False).input_ids 
    if isinstance(src_sent, str):
        src_sent = re.sub(r'\s+', ' ', src_sent) # remove redundant spaces in src_sent
        num_masks = src_sent.count('<mask>') # count number of masks
        tgt_seq = re.sub(r'<src>.*<temp_t>', r'</s><s> <temp_t>', src_sent) # replace the source part with </s><s> <temp_t>
        encoder_input_ids = tok(src_sent, return_tensors='pt').input_ids.to(model.device) # tokenize the source sentence and 
        # move the input_ids to the same device as the model
        tgt_seq_input_ids = tok(tgt_seq, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device) # prepare the input_ids for the decoder
        mask_id = tok('<mask>', add_special_tokens=False).input_ids[0] # get the tokenizer id of <mask>
    elif isinstance(src_sent, Tensor):
        mask_id = tok.mask_token_id
        src_sent = src_sent.tolist()
        num_masks = src_sent.count(tok.mask_token_id) # count number of masks
        encoder_input_ids = Tensor(src_sent).to(model.device).unsqueeze(0).long()
        tgt_seq_input_ids = Tensor(tok('</s> <s>', add_special_tokens=False).input_ids + src_sent[src_sent.index(tok.convert_tokens_to_ids('<temp_t>')):]).unsqueeze(0).long().to(model.device)

    while num_masks != 0:  # loop until all masks are replaced with the generated tokens
        if debug:
            print('==' * 30)
        # print('current tgt_seq_input', tok.batch_decode(tgt_seq_input_ids)[0])
        mask_idx = tgt_seq_input_ids[0].tolist().index(mask_id)  # get the index of the first mask
        decoder_input_ids = tgt_seq_input_ids[:, :mask_idx].to(model.device)  # get the decoder input ids by truncating the tgt_seq_input_ids to the first mask
        if debug:
            print("decoder_inp:", tok.batch_decode(decoder_input_ids)[0])
        generated_ids = model.generate(inputs=encoder_input_ids, 
            decoder_input_ids=decoder_input_ids, eos_token_id=right_bracket_id if num_masks > 1 else tok.eos_token_id,
            **generation_kwargs)
        # extract the REAL generated ids
        mask_corresponding_ids = generated_ids[0][decoder_input_ids.shape[-1]:]
        # remove brackets from generated ids to prevent error forming trees
        # original_size = mask_corresponding_ids.size(-1)
        for each in bracket_ids:
            mask_corresponding_ids = mask_corresponding_ids[mask_corresponding_ids != each]
        # if mask_corresponding_ids.size(-1) != original_size:
        #     breakpoint()
        generated_ids = torch.cat([
            generated_ids[0][:decoder_input_ids.size(-1)],
            mask_corresponding_ids,
            Tensor(right_bracket_id if num_masks > 1 else []).long().to(generated_ids.device)
        ], dim=0).unsqueeze(0)
        # generate the tokens corresponding to the first mask
        generated_seq = tok.batch_decode(generated_ids)[0]   # decode the generated ids
        if debug:
            print("decoder_out:", generated_seq)
        num_masks -= 1  # decrease the number of masks by 1
        tgt_seq_input_ids = torch.cat([generated_ids, tgt_seq_input_ids[:, mask_idx + 2:]], dim=1)  
        # concatenate the generated ids with the rest of the tgt_seq_input_ids
    # print(*zip(results.tolist()[0], tok.convert_ids_to_tokens(results[0])), sep='\n')
    return generated_ids

def val_test(opt: Namespace, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, data_mode: str, current_epoch: Union[int, str], loader: DataLoader, generation_args: Dict[int, str], constraints: List[Constraint]):
    """Common implementation for val and test. 
    Parameters:
        opt: arguments parsed by argument parser.  
        current_epoch: current_epoch
        loader: validation/test dataloader
    Returns:
        generation results and metrics"""
    logger.info(f'evaluating at epoch {current_epoch} (mode {data_mode})...')
    sleep(0.05)
    can_show_pbar = not opt.generation_debug and (not dist.is_initialized() or dist.get_rank() == 0)
    if can_show_pbar:
        tqdm_val_loader = tqdm(total=len(loader), mininterval=2.0, ncols=100)
    extra_ids = [tokenizer.convert_tokens_to_ids(f'<extra_id_{i}>') for i in range(100)]

    model.eval()
    if opt.generation_debug:
        # prepare non-terminal token-ids for logit NT-token visualization
        NT_tokens = pkl.load(open('./data/ParaNMT50m_original/NT.pkl', 'rb'))
        NT_token_ids = [tokenizer(each, add_special_tokens=False).input_ids for each in NT_tokens]
        whitechar_idx = tokenizer.convert_tokens_to_ids(chr(0x2581))
        for idx, each in enumerate(NT_token_ids): # avoid NT-tokens which are always considered as subtokens (like `,`, '.', ':')
            if len(each) == 2 and each[0] == whitechar_idx:
                NT_token_ids[idx] = [each[1]]
        # check if NT_tokens are split into subwords
        NT_token_id_lenghts = [*map(len, NT_token_ids)]
        assert (max_NT_length := max(map(len, NT_token_ids))) == 1, \
            f'error, NT_token `{NT_tokens[NT_token_id_lenghts.index(max_NT_length)]}` ' \
            f'is split into subtokens: {tokenizer.convert_ids_to_tokens(NT_token_ids[NT_token_id_lenghts.index(max_NT_length)])}'
        NT_token_ids = [each[0] for each in NT_token_ids]
        sequences_and_attentions = []
    # 创建输出文件（epoch.txt和epoch_raw.txt）
    if opt.output_dir is not None:
        results_dir = os.path.join(opt.output_dir, 'results', data_mode)
        if dist.is_available():
            # output_filepath = os.path.join(results_dir, (f'{current_epoch}_chunk{dist.get_rank()}.txt' if dist.is_initialized() else f'{current_epoch}_chunk0.txt')) 
            # open(output_filepath, 'w')
            output_filepath_raw = os.path.join(results_dir, (f'{current_epoch}_chunk{dist.get_rank()}_raw.txt' if dist.is_initialized() else f'{current_epoch}_chunk0_raw.txt'))
            open(output_filepath_raw, 'w')
        else:
            # output_filepath = os.path.join(results_dir, f'{current_epoch}_chunk0.txt')
            # open(output_filepath, 'w')
            output_filepath_raw = os.path.join(results_dir, f'{current_epoch}_chunk0_raw.txt')
            open(output_filepath_raw, 'w')

    # all_inputs = []
    # all_predicted_texts = []
    # all_label_texts = []
    # 进行inference并将结果存放到 all_inputs, all_predicted_texts和all_label_texts中
    # all_generated_seqs = []

    for data_idx, batch in enumerate(loader):
        # tokenized_inputs = tokenizer(inputs, **tokenizer_kwargs).to(device)
        if opt.model_type in ['t5', 'bart']:
            # prev_time = time()
            generated_seqs = model.generate(
                **batch.to(model.device), **generation_args, constraints=constraints if constraints else None, 
                output_attentions=opt.generation_debug, 
                output_scores=opt.generation_debug, 
                return_dict_in_generate=opt.generation_debug)
            # breakpoint()

            if opt.generation_debug:
                # generated_seqs:
                # sequences: Tensor[batch_size, max_generated_length_among_this_batch]
                # scores: Tuple {
                #   length = max_generated_length_among_this_batch - 1 (next token scores)
                #   element = Tensor[batch_size, vocab_size]
                # }
                # cross_attentions: Tuple {
                #   [ max_generated_length_among_this_batch - 1, num_layers, Tensor[batch_size, num_heads, decoder_seq_length, encoder_seq_length]]
                # }
                sequences_and_attentions.append({'token': generated_seqs['sequences'], 'attention': generated_seqs['cross_attentions']})
                # # visualize next token scores for NT tokens
                # for token_idx in range(len(generated_seqs.scores)):
                #     # tokens before the current visualized token
                #     print(f'context: {tokenizer.decode(generated_seqs.sequences[0][: token_idx + 1])}')
                #     print(f'token: {tokenizer.decode(generated_seqs.sequences[0][token_idx + 1])}')
                #     current_score = generated_seqs.scores[token_idx][0].tolist()
                #     tokenidx2score = sorted(enumerate(current_score), key=lambda x: x[1], reverse=True)
                #     NT_tokenidx2score = [*filter(lambda x: x[0] in NT_token_ids, tokenidx2score)]

                #     print('top-10 nt-tokens:')
                #     # for _, (NT_tokenidx, score) in zip(range(20), NT_tokenidx2score):
                #     for NT_tokenidx, score in NT_tokenidx2score:
                #         print(f'{tokenizer.decode(NT_tokenidx)} {score:.6f}')

                # breakpoint()
                # generated_seqs.copy
                # attention [Position_id(tuple), layer(tuple), batch_size * num_beams, head, key_value_id]
                # print(tokenizer.batch_decode(generated_seqs))
            # generated_seq = generated_seqs[0].tolist()
            # src = batch.input_ids[0].tolist()
            # try:
            #     src_extra_ids = [each for each in src if each in extra_ids]
            #     begin_idx = generated_seq.index(src_extra_ids[0])
            #     ending_extra_id = extra_ids[min(extra_ids.index(src_extra_ids[-1]), 99)]
            #     end_idx = generated_seq.index(ending_extra_id)
            #     generated_seq = [each for each in generated_seq[begin_idx + 1: end_idx] if each not in extra_ids]
            # except ValueError as e1:
            #     generated_extra_ids = [each for each in generated_seq if each in extra_ids]
            #     logger.warning(f'Found value error')
            #     breakpoint()
            #     logger.warning(f'generated_seq: {tokenizer.batch_decode(generated_seqs)}')
            #     logger.warning(f'src: {tokenizer.decode(src)}')
            #     logger.warning(f'src_extra_ids: {[tokenizer.convert_ids_to_tokens(each) for each in src_extra_ids]}')
            #     logger.warning(f'ending_extra_id: {ending_extra_id}')
            #     generated_seq = [each for each in \
            #         generated_seq[
            #             generated_seq.index(generated_extra_ids[0]) + 1:# \
            #             # generated_seq.index(generated_extra_ids[-1])
            #         ] if each not in extra_ids and each != tokenizer.eos_token_id]
            # except IndexError as e:
            #     generated_extra_ids = [each for each in generated_seq if each in extra_ids]
            #     logger.warning(f'Found index error')
            #     breakpoint()
            #     logger.warning(f'generated_seq: {tokenizer.batch_decode(generated_seqs)}')
            #     logger.warning(f'src: {tokenizer.decode(src)}')
            #     logger.warning(f'src_extra_ids: {[tokenizer.convert_ids_to_tokens(each) for each in src_extra_ids]}')
            #     logger.warning(f'ending_extra_id: {ending_extra_id}')
            #     generated_seq = [each for each in \
            #         generated_seq[
            #             generated_seq.index(generated_extra_ids[0]) + 1:# \
            #             # generated_seq.index(generated_extra_ids[-1])
            #         ] if each not in extra_ids and each != tokenizer.eos_token_id]

            # print('time taken(t5 generate):', time() - prev_time)
        # elif opt.model_type == 'bart':
        #     # prev_time = time()
        #     generated_seqs = teacher_forcing_generate(model, tokenizer, batch['input_ids'][0], **generation_args)
        #     generated_seqs_decoded = tokenizer.batch_decode(generated_seqs)[0]
        #     tgt_tree = re.search(r'<temp_t>(.*)<temp>', generated_seqs_decoded).group(1)
        #     try:
        #         generated_seq = ' '.join(Tree.fromstring(tgt_tree).get_all_leaves())
        #     except Exception as e:
        #         logger.warning(f'Error occurred during processing tgt_tree `{tgt_tree}`')#'attempting to add right bracket...')
        #         breakpoint()
        #         tgt_tree = re.sub(r'[()]', '', tgt_tree)
        #         tgt_tree = re.sub(r' +', ' ', tgt_tree)
        #         tgt_tree = tokenizer(tgt_tree, add_special_tokens=False).input_ids
        #         generated_seq = tokenizer.decode([each for each in tgt_tree if each not in tokenizer.NT_idx_set])
                # print(end='')

            # print('time taken(bart generate):', time() - prev_time)
        # all_inputs.extend(inputs)
        # all_generated_seqs.append(generated_seq)
        # if (data_idx + 1) % 20 == 0 or (data_idx + 1) == len(loader):
        #     logger.info(f'evaluation: ({data_idx + 1}/{len(loader)}) batches done')

        if can_show_pbar:
            tqdm_val_loader.update()
        # with open(output_filepath, 'a') as f:
        #     f.write(tokenizer.decode(generated_seq) + '\n' if isinstance(generated_seq, list) else generated_seq + '\n')
        with open(output_filepath_raw, 'a') as f:
            file_content_to_write = '\n'.join([each.replace('\n', '').replace('<pad>', '') for each in tokenizer.batch_decode(generated_seqs)]) + '\n'
            f.write(file_content_to_write)
        # generated_seqs = ([form_output_sentence(each) for each in generated_seqs])
        # all_label_texts.extend(labels)
    
    if opt.generation_debug:
        pkl.dump(sequences_and_attentions, open(f'./Paraphrase/sequences_and_attentions.pkl', 'wb'))

    # total_bleu, total_rouge1, total_rouge2, total_rougel, total_meteor = 0., 0., 0., 0., 0.
    # print('calculating metrics and writing results...')
    # 遍历all_inputs, all_predicted_texts和all_label_texts中，将inference结果进行处理并写入文件
        # for src, generated, annotated in zip(inputs, generated_seqs, labels):
        #     if opt.output_dir is not None:
        #         with open(output_filepath_raw, 'a') as f:
        #             f.write(f'source: {src}\n')
        #             f.write('\tpredicted: ' + generated.replace('<pad>', '') + '\n')
        #             f.write('\tannotated: ' + annotated + '\n')
        #     # 将src（结构形如"原语句+展平句法树 </s> 目标句法树+<extra_ids> </s>"）转变为原语句部分
        #     if opt.model_type == 't5':
        #         src = re.sub('<extra_id_[0-9]+>', '', src)
        #         src = re.sub('</s>.*</s>', '', src)
        #     elif opt.model_type == 'bart':
        #         src = re.sub('<src_t>.*<temp_t>', '', src)
        #     else:
        #         raise NotImplementedError

        #     res = []
        #     for element in src.split(' '):
        #         if not element.startswith('(') and not element.endswith(')') and element not in tree_nonleaf_set:
        #             res.append(element)
                    
        #         # print(src)
        #     src = ' '.join(res)
        #     if opt.model_type == 't5':
        #         extra_ids = list(map(int, re.findall('<extra_id_([0-9]+)>', generated)))
        #         # print(generated)
        #         # 将generated截取第一个<extra_id>到最后一个<extra_id>之间的部分并拼接
        #         if len(extra_ids) >= 2:
        #             generated_match = re.match(f'<extra_id_{extra_ids[0]}>(.*)<extra_id_{extra_ids[-1]}>', generated)
        #             if generated_match is not None:
        #                 generated = generated_match.group(1)
        #         generated = re.sub(
        #             '<extra_id_[0-9]+>',
        #             '',
        #             generated).strip()
            
        #     elif opt.model_type == 'bart':
        #         generated = preprocess(generated)
        #         res = []
        #         generated = re.sub(r'<src_t>.*<temp_t>', '', generated)
        #         for element in generated.split(' '):
        #             if not element.startswith('(') and not element.endswith(')') and element not in tree_nonleaf_set:
        #                 res.append(element)
                        
        #             # print(src)
        #         generated = ' '.join(res)

        #     generated = re.sub(' +', ' ', generated)

        # total_bleu += bleu.compute(predictions=[generated], references=[annotated])['bleu']
        # rouges = rouge.compute(predictions=[generated], references=[annotated])
        # total_rouge1 += rouges['rouge1']
        # total_rouge2 += rouges['rouge2']
        # total_rougel += rouges['rougel']
        # total_meteor += meteor.compute(predictions=[generated], references=[annotated])
            # if opt.output_dir is not None:
            #     with open(output_filepath, 'a') as f:
            #         f.write(f'source: {src}\n')
            #         f.write('\tpredicted: ' + generated + '\n')
            #         f.write('\tannotated: ' + annotated + '\n')

    # 存储模型（如果是训练过程中的cross validation的话）
    # if opt.output_dir and not opt.test_only:
    #     model_serialization_path = os.path.join(checkpoint_dir, f'{epoch}')
    #     if not os.path.exists(model_serialization_path):
    #         os.mkdir(model_serialization_path)
    #     tokenizer.save_pretrained(model_serialization_path)
    #     model.save_pretrained(model_serialization_path)
