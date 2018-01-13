import os, sys, time
import logging
import numpy as np


## global
logger = logging.getLogger('logger')

def setConfig(para):
    config = {
        'exeFile': os.path.basename(sys.argv[0]),  
        'workPath': os.path.abspath('.'),
        'dataPath': os.path.abspath(para['dataPath']),
        'logFile': para['dataPath']+os.path.basename(sys.argv[0]) + '.log'
    }

    # delete old log file
    if os.path.exists(config['logFile']):
        os.remove(config['logFile'])
    # add result folder
    if not os.path.exists(para['outPath']):
        os.makedirs(para['outPath'])

    # set up logger to record runtime info
    if para['debugMode']:  
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO) 

    # log to console
    cmdhandler = logging.StreamHandler()  
    cmdhandler.setLevel(logging.DEBUG)       
    formatter = logging.Formatter('%(asctime)s (pid-%(process)d): %(message)s')
    cmdhandler.setFormatter(formatter)
    logger.addHandler(cmdhandler)

    # log to file
    if para['saveLog']:
        filehandler = logging.FileHandler(config['logFile']) 
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(formatter)       
        logger.addHandler(filehandler)  
    
    logger.info('==========================================')
    logger.info('configs as follows:')
    config.update(para)

    for name in config:
        if type(config[name]) is np.ndarray:
            logger.info('%s = [%s]'%(name, ', '.join(format(s, '.2f') for s in config[name])))
        else:
            logger.info('%s = %s'%(name, config[name]))
    
    # set print format
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


if __name__=="__main__":
    para={
        'dataPath': './data/',
        'outPath': './result/',
        'saveLog': True, # whether to save log into file
        'debugMode': False, # whether to record the debug info
    }
    setConfig(para)