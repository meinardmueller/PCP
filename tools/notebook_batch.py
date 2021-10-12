"""
Author: Frank Zalkow, International Audio Laboratories Erlangen
This file is part of the PCP Notebooks (https://www.audiolabs-erlangen.de/PCP)
"""

import argparse
import os


IGNORE_PRAEFIX = 'Dev_'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch clean, execute and html-export notebooks')
    parser.add_argument('directory', help='directory, which is traversed recursively')
    parser.add_argument('--mode', help='clean, execute, html (default: html)', default='html')
    args = parser.parse_args()

    assert args.mode in ['clean', 'execute', 'html']

    if args.mode == 'clean':
        opt = '--to notebook --ClearOutputPreprocessor.enabled=True --clear-output'
    if args.mode == 'execute':
        opt = '--ExecutePreprocessor.timeout=3600 --to notebook --execute --inplace'
    if args.mode == 'html':
        # opt = '--to html --template classic'
        opt = '--to html'

    for dirpath, dirnames, filenames in os.walk(args.directory):

        # ignore .ipynb_checkpoints directories
        if '.ipynb_checkpoints' in os.path.normpath(dirpath).split(os.path.sep):
            continue

        # else process all *.ipynb files
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.ipynb' and filename[:len(IGNORE_PRAEFIX)] != IGNORE_PRAEFIX:
                fn_ipynb = os.path.join(dirpath, filename)

                cmd = 'jupyter nbconvert {options} {infile}'.format(options=opt, infile=fn_ipynb)
                print(cmd)
                os.system(cmd)
