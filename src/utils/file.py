#!./env python
import os
import sys
import shutil
import re
import errno
import torch

__all__ = ['check_path', 'save_checkpoint']

def save_checkpoint(epoch, net, optimizer, scheduler=None, path='.', filename='checkpoint.pth.tar'):
    filepath = os.path.join(path, filename)
    state = {'epoch': epoch + 1,
             'state_dict': net.state_dict(),
             'optimizer' : optimizer.state_dict(),
            }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    torch.save(state, filepath)


def copyanything(path, src, dst):
    path_ = os.path.join(path, src)
    if os.path.isdir(path_):
        shutil.copytree(path_, os.path.join(dst, src))
    else:
        shutil.copy(path_, dst)

    # except OSError as exc: # python >2.5
    #     if exc.errno == errno.ENOTDIR:
    #         shutil.copy(src, dst)
    #     else: raise

def get_not_ext(exts, path='.'):
    files = []
    for f in os.listdir(path):
        if not any([f.endswith(ext) for ext in exts]):
            files.append(f)
    return files

def check_path(path, config):
    if not os.path.exists(path):
        os.mkdir(path)
        return path

    option = input('Path %s already exists. Delete[d], Rename[r], Abort[a], Continue[c], Terminate[*]? ' % path)
    if option.lower() == 'd':
        shutil.rmtree(path)
        os.mkdir(path)
        return path

    if option.lower() == 'r':
        sufs = re.findall(r'-(\d+)$', path)
        if not sufs:
            path = path + '-1'
        else:
            i = int(sufs[0]) + 1
            path = re.sub(r'-(\d+)$', '-%i' %i, path)
        return check_path(path, config)

    if option.lower() == 'a':
        sys.exit(1)

    if option.lower() == 'c':
        assert(config['resume']), 'resume is not set in config!'
        # continue / resume
        save_dir = check_path(os.path.join(path, 'old'), config)
        exts = ['.pt', '.tar', 'old', '.npy']
        for f in get_not_ext(exts, path=path):
            if f.startswith('old'):
                continue
            if not f.endswith('.txt'):
                shutil.move(os.path.join(path, f), save_dir)
            else:
                copyanything(path, f, save_dir)
        return path

    sys.exit(2)

