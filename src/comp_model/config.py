'''
#  =======================================================
#  = = = = = = = = = CONFIG FILE = = = = = = = = = = = = =
#  =======================================================
'''

# =============================================================================
# MODULES
# =============================================================================
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# =============================================================================
# ALLIASES
# =============================================================================
op=os.path
see=os.listdir
join=op.join
exists=op.exists
make=os.makedirs
real=op.realpath


# =============================================================================
# CONSTRUCTIONS
# =============================================================================
constructions=['pp_syntax','objrc_syntax','pp_semantics', ]
# =============================================================================
# EFFECTS
# =============================================================================
first_order_effects =[
    'main_effect_of_violation',
    # 'main_effect_of_congruency',
    # 'main_effect_of_transition'
    ]
second_order_effects=['violation_congruency_interaction']


# =============================================================================
# DIRECTORIES
# =============================================================================
root=os.path.realpath(join(os.sep,'..'))





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





















