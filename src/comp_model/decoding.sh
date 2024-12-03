# Syntax-PP (number)
python decoding.py --queries "feature=='number' & condition in ['GSLS', 'GSLD'] & structure=='pp'" "condition in ['GDLD', 'GDLS'] & structure=='pp'"
python decoding.py --queries "feature=='number' & condition in ['GSLS', 'GDLS'] & structure=='pp'" "condition in ['GSLD', 'GDLD'] & structure=='pp'"
python decoding.py --queries "feature=='number' & condition in ['GSLS', 'GDLD'] & structure=='pp'" "condition in ['GSLD', 'GDLS'] & structure=='pp'"
# ObjRel-PP (number)
python decoding.py --queries "feature=='number' & condition in ['GSLS', 'GSLD'] & structure=='obj'" "condition in ['GDLD', 'GDLS'] & structure=='obj'"
python decoding.py --queries "feature=='number' & condition in ['GSLS', 'GDLS'] & structure=='obj'" "condition in ['GSLD', 'GDLD'] & structure=='obj'"
python decoding.py --queries "feature=='number' & condition in ['GSLS', 'GDLD'] & structure=='obj'" "condition in ['GSLD', 'GDLS'] & structure=='obj'"
# Syntax-PP (animcay)
python decoding.py --queries "feature=='animacy' & condition in ['GSLS', 'GSLD'] & structure=='pp'" "condition in ['GDLD', 'GDLS'] & structure=='pp'"
python decoding.py --queries "feature=='animacy' & condition in ['GSLS', 'GDLS'] & structure=='pp'" "condition in ['GSLD', 'GDLD'] & structure=='pp'"
python decoding.py --queries "feature=='animacy' & condition in ['GSLS', 'GDLD'] & structure=='pp'" "condition in ['GSLD', 'GDLS'] & structure=='pp'"
