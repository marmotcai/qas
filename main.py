import os
import sys
import time
import getopt
import schedule
import trendanalysis as ta
from trendanalysis import global_obj
from trendanalysis.core import manager as g_man
from trendanalysis.core import data_manager as g_dm
from trendanalysis.utils import tools as my_tools
import update

################################################################################

def schedule_analysis(at, cmd, args):
    if len(at) > 0 and at[0] in ("seconds"):
        schedule.every(int(at[1])).seconds.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("minutes"):
        schedule.every(int(at[1])).minutes.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("hour"):
        schedule.every(int(at[1])).hour.do(main, cmd, args)

    if len(at) > 0 and at[0] in ("day"):
        schedule.every().day.at(at[1]).do(main, cmd, args)

    ##############################################################

    if len(at) > 0 and at[0] in ("monday"):
        schedule.every().monday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("tuesday"):
        schedule.every().tuesday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("wednesday"):
        schedule.every().wednesday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("thursday"):
        schedule.every().thursday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("friday"):
        schedule.every().friday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("saturday"):
        schedule.every().saturday.at(at[1]).do(main, cmd, args)
    if len(at) > 0 and at[0] in ("sunday"):
        schedule.every().sunday.at(at[1]).do(main, cmd, args)

def service(filename = ""):
    g = ta.g
    if (len(filename) > 0):
        g = global_obj.Global(filename)

    g.log.info("start service...")
    for index in range(0, len(g.schedules)):
        item = g.schedules[index]
        at = item['at'].split(",")
        cmdlst = item['cmd'].split(",")
        if len(cmdlst) > 1:
            cmd = cmdlst[0]
            args = cmdlst[1].split(" ")
        else:
            cmd = "python"
            args = cmdlst[0].split(" ")

        ta.g.log.info("add schedule: " + " at: " + item['at'] + " cmd: " + item['cmd'])

        schedule_analysis(at, cmd, args)

    while True:
        schedule.run_pending()
        time.sleep(5)

def usage():
    print("-h --help,           Display this help and exit")
    print("-v --version,        Print version infomation")
    print("-u --update,         update new ver")
    print("-i --initialize,    initialize")
    print("-d --daemon,         daemon mode")
    print("-t --test,           Test mode (e.g:-t gpu)")
    print("-d --download,       Download data (e.g:-d inx=inx_code.csv)")
    print("-m --modeling,       Training data build model")
    print("-e --evaluation,     Evaluation model")

def main(cmd, argv):
    if len(cmd) > 0: # 用进程方式启动
        my_tools.process(cmd, argv).start()
        return
    try:
        options, args = getopt.getopt(argv, "hvuiss:t:d:m:p:e:", ["help", "version", "update", "initialize", "service", "service=", "test=", "download=", "modeling=", "evaluation=", "predict="])
    except getopt.GetoptError:
        sys.exit()

    for name, value in options:
        if name in ("-h", "--help"):
            usage()
        if name in ("-v", "--version"):
            my_tools.print_sys_info()
            ta.g.print_current_env_nformation()
        if name in ("-u", "--update"):
            update.main(argv)
        if name in ("-i", "--initialize"):
            g_man.main(argv)
        if name in ("-s", "--service"):
            platform = my_tools.UsePlatform()
            if ("windows" == platform.lower()):
                print("windows platform is not suppor daemon mode and run service mode")
                service("")
            else:
                os.system(sys.executable + " daemon_main.py " + value)
        if name in ("-t", "--test"):
            g_man.test(value)
        if name in ("-d", "--download"):
            g_dm.main(argv)
        if name in ("-m", "--modeling"):
            g_man.main(argv)
        if name in ("-p", "--predict"):
            g_man.main(argv)
        if name in ("-e", "--evaluation"):
            g_man.evaluation(value)

if __name__ == '__main__':
    main("", sys.argv[1:])
    sys.exit()