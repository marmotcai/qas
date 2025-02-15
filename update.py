
import sys
import os
import getopt
from git import Repo
import trendanalysis as ta

# 本模块需要先执行以下命令,并登录一次
# git config --global credential.helper store

default_giturl = 'github.com/marmotcai/qas.git'
default_gitdir = '/home/x/workspace/qas'

def update(githuburl):

    ta.g.log.info("start git pull..")

    repo = Repo(os.path.dirname(__file__))

    # 获取master最新版本的hexsha值
    head = repo.head
    master = head.reference
    log = master.log()
    newhexsha = (log[-1].newhexsha)
    ta.g.log.info("new hexsha: " + newhexsha)

    files = []
    origin = repo.remotes.origin
    fetch_info = origin.pull()
    for single_fetch_info in fetch_info:
        for diff in single_fetch_info.commit.diff(
                single_fetch_info.old_commit):
            ta.g.log.info("Found diff: " + str(diff))
            if not diff.a_blob is None:
                if not diff.a_blob.path in files:
                    ta.g.log.info("pull file: " + diff.a_blob.path)
                    files.append(diff.a_blob.path)

    ta.g.log.info("stop git pull (file count: {})".format(str(len(files))))

def main(argv):
    try:
        options, args = getopt.getopt(argv, "u", ["update"])
    except getopt.GetoptError:
        sys.exit()

    for name, value in options:
        if name in ("-u", "--update"):
            update(value)

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit()