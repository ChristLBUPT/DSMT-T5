from subprocess import Popen, PIPE
import sys
from time import time, sleep
import os
import re

def gracefully_shutdown(pid: int, wait_time: float = 5):
    print(f'wait for {wait_time}s before shutting down all subprocesses')
    # pstrees = Popen(['pstree', '-apn', f'{pid}'], stdout=PIPE, stderr=PIPE).communicate()[0].decode()
    psef_process = Popen(['ps', '-ef'], stdout=PIPE)
    grep_process = Popen(['grep', 'para_pretrain_multitask.py '], stdin=psef_process.stdout, stdout=PIPE)
    awk_process = Popen(['awk', "{print $2}"], stdin=grep_process.stdout, stdout=PIPE)
    pids = [int(each) for each in awk_process.communicate()[0].decode().split('\n') if each]

    # psef_found_pids = Popen(['ps', '-ef', '|', 'grep', f'"para_pretrain_multitask.py {" ".join(sys.argv[1:])}"', '|', 'awk', "'{print $2}"] , stdout=PIPE, stderr=PIPE).communicate()[0].decode()
    # for line in pstrees.split('\n'):
    #     pids.extend(re.findall(r'\d+', line))
    
    print(f'found processes {", ".join(map(str, pids))}')
    sleep(wait_time)
    for this_pid in pids:
        print(f'killing process {this_pid}..')
        os.system(f'kill -9 {this_pid}')


if __name__ == "__main__":
    try:
        ret_value = -1
        failed = False
        while ret_value != 0:
            start_time = time()
            command = [
                sys.executable, '-m', 'accelerate.commands.launch', 'para_pretrain_multitask.py'
            ] + sys.argv[1:]
            if failed and '--resume' not in command:
                command.append('--resume')
            print(f'running command `{" ".join(command)}`')
            sp = Popen(command, stdout=sys.stdout, stderr=sys.stderr)
            sp.communicate()
            elapse = time() - start_time
            if elapse < 30:
                print(f'run for less than 30({elapse:.2f}) seconds, terminating...')
                sys.exit(1)

            ret_value = sp.returncode
            if ret_value != 0:
                print(f'get non-zero return value, re-operating...')
                failed = True
                gracefully_shutdown(sp.pid)
                # os.system(f'ps -ef | grep {filter()}')
    except KeyboardInterrupt:
        print(f'attempting gracefully shutdown...')
        # os.system(f'ps -ef | grep {sp.pid} | xargs kill -9')
        gracefully_shutdown(sp.pid)