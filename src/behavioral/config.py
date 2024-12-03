'''
#  =======================================================
#  = = = = = = = = = CONFIG FILE = = = = = = = = = = = = =
#  =======================================================
#  This version of the config file is specifically for the 
#  project lang_loc_glob by Christos. To see the original 
#  config go to:
#  https://github.com/brainthemind/CogBrainDyn_MEG_Pipeline
#  Configuration parameters for the current study.
'''

# IMPORT MODULES ######################
import sys
sys.path.append('../')
import os
import numpy as np
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
#######################################

# =============================================================================
# ALLIASES
# =============================================================================
op=os.path
see=os.listdir
join=op.join
exists=op.exists
make=os.makedirs

###############################################################################
plot = False
autoreject=False

# =============================================================================
# PROJECT NAME
# =============================================================================
project_name ='LocalGlobal@ENGLISH'


# =============================================================================
# CONSTRUCTIONS
# =============================================================================
constructions=['pp_syntax','objrc_syntax','pp_semantics', ]

# =============================================================================
# DIRECTORIES
# =============================================================================
root=os.path.join(os.sep,'media','cz257680','Transcend2',)
# root=os.path.join(os.sep,'neurospin','unicog','protocols',)


# =============================================================================
# SUBJECTS
# =============================================================================

subjects_list = [
            # # 'B01',
                'P01',
                'P02',
                'S01',
                'S02',
                'S03',
                'S04',
                'S05',
                'S06',
                'S07',
                'S08',
                'S09',
                'S10',            
                'S11',
                'S12',
                'S14',
                'S15',
              'ICM01',
              'ICM02',
              'ICM03',
              'ICM04',
            # # 'ICM05',
              'ICM06',
              'ICM07',
            ]
