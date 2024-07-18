import sys
if __name__ == '__main__':
    run_type = sys.argv.pop(1)
    if run_type == 'para_train':
        from Paraphrase.para_train import main as para_train_main
        para_train_main()
    elif run_type == 'post_process':
        from Paraphrase.post_process import main as post_process_main
        post_process_main('./runs/')
    else:
        raise ValueError(f'error, unsupported run_type: `{run_type}`')