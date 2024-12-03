'''
WHERE THE FUCK IS MY MISTAKE?
'''
# =============================================================================
# MODULES
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as c

# =============================================================================
# LOAD DATA
# =============================================================================
def load_data(subject):
    print(f'Loading data for subject: {subject}')
    path2data=c.join(c.root,c.project_name,'Data',subject,'Behavioral_Responses')
    files=[pd.read_csv(c.join(path2data,f), sep='\t') for f in c.see(path2data)]
    dataset=pd.concat(files)
    # drop the index
    dataset.reset_index(drop=True, inplace=True)
    
    ## CURATE THE DATAFRAME AND ADD FEATURES
    # drop the first column
    del dataset['Var1']
    # remove the NaNs (no responses from the subjects)
    dataset = dataset[dataset['RT'].notna()]
    dataset['category']=dataset.condition.apply(lambda x: x.split('_')[-1])
    dataset['number']=dataset.condition.apply(lambda x: x.split('_')[-2])
    is_objrc=dataset.condition.apply(lambda x: x.split('_')[1]).apply(lambda y:'C' in y)
    dataset['structure']=''
    dataset['structure'][is_objrc]='obj'
    dataset['structure'][~is_objrc]='pp'
    dataset['condition']=dataset.condition.apply(lambda x: x.split('_')[0])
    # Now, add the main effects, violation, congruency and transition
    dataset['congruency']=''
    congruent = ['GSLS', 'GDLD']
    incongruent=['GSLD', 'GDLS']
    dataset['congruency'][dataset['condition'].isin(congruent)]='yes'
    dataset['congruency'][dataset['condition'].isin(incongruent)]='no'
    dataset['response']=''
    correct=['TP','TN']
    false=['FP','FN']
    dataset['response'][dataset['Behavioral'].isin(correct)]='correct'
    dataset['response'][dataset['Behavioral'].isin(false)]='false'    
    dataset['violation']=''
    dataset['violation'][dataset['violIndex']==1]='yes'
    dataset['violation'][dataset['violIndex']==0]='no'
    
    dataset = dataset.rename(columns={'category': 'feature'})
    dataset['feature']=dataset['feature'].replace({'syntactic': 'number', 'semantic': 'animacy',})
    # Add the transition (will clean up later if needed)

    dataset['linear_interference']=''
    # PP
    dataset['linear_interference'][(dataset.condition=='GSLS') & (dataset.structure=='pp')]=0
    dataset['linear_interference'][(dataset.condition=='GSLD') & (dataset.structure=='pp')]=1
    dataset['linear_interference'][(dataset.condition=='GDLS') & (dataset.structure=='pp')]=0
    dataset['linear_interference'][(dataset.condition=='GDLD') & (dataset.structure=='pp')]=1

    # OBJ
    dataset['linear_interference'][(dataset.condition=='GSLS') & (dataset.structure=='obj')]=0
    dataset['linear_interference'][(dataset.condition=='GSLD') & (dataset.structure=='obj')]=0
    dataset['linear_interference'][(dataset.condition=='GDLS') & (dataset.structure=='obj')]=1
    dataset['linear_interference'][(dataset.condition=='GDLD') & (dataset.structure=='obj')]=1


    return dataset




def tranform_dataframe_for_ANOVA(count,df):

    df['violation']=df['violation'].map({'yes':1,'no':0})
    df['congruency']=df['congruency'].map({'yes':1,'no':0})

    # keep only the features of interest (error, violation, congruency)
    df=df[['response','violation','linear_interference','congruency','number','structure','feature','sentence']]
    df = df.astype({'violation':'int', 'linear_interference':'int', 'congruency':'int'})

    # return the dataframe
    
    return df




def get_the_error_per_unique_distribution(df, count):

    # First, get the number of unique combinations (should be == Nconditions (24))
    unique=df.drop(['response','sentence'], axis=1).drop_duplicates()
    if not unique.shape[0]==24:
        raise ValueError('Problem with #conditions')
    
    # get the total number of trials
    n_trials_total=df.shape[0]

    # initialize a collector to keep the new dfs
    false_responses, total_trials, trials_per_construction_and_feature, trials_per_unique_combination=([] for i in range(0,4))
    # Now, loop through the unique combinations and calculate the error rate in the original dataframe
    for condition in range(unique.shape[0]):

        selected_values= unique.iloc[condition].to_dict()
        # Now loop through the rows and columns of the dict
        q = f"structure == '{selected_values['structure']}' &\
            feature == '{selected_values['feature']}'"
        # get the total number of trials to calculate the error-rate
        ntrials_per_construction=df.query(q).shape[0]
                 
        # QUERY 
        isolated=df[df.violation==selected_values['violation']]         
        isolated=isolated[isolated.congruency==selected_values['congruency']]      
        isolated=isolated[isolated.linear_interference==selected_values['linear_interference']]              
        isolated=isolated[isolated.number==selected_values['number']]         
        isolated=isolated[isolated.structure==selected_values['structure']]         
        isolated=isolated[isolated.feature==selected_values['feature']]



        # calcuate the error-rate for this instance of features
        false=isolated[isolated.response=='false'].shape[0]
        # Append to the collectors
        false_responses.append(false)
        # Total #trials for this subject (e.g 480)
        total_trials.append(n_trials_total)
        # Total #trials that correspond to the construction (e.g: PP-Number: e.g 160)
        trials_per_construction_and_feature.append(ntrials_per_construction)
        # Total #trials that correspond to this specific set of conditions (e.g: Violation:0, Congruency:1, Interference:0)
        trials_per_unique_combination.append(isolated.shape[0])


    unique['false_responses']=false_responses
    unique['total_trials']=total_trials
    unique['trials_per_construction']=trials_per_construction_and_feature
    unique['trials_per_combination']=trials_per_unique_combination
    
    unique.index=[count]*unique.shape[0]
    
    return unique



def cacl_accuracy(data):
    correct=data[data.response=='correct'].shape[0]
    total=data.shape[0]
    
    accuracy=round((correct/total)*1e2,2)
    return accuracy

 
def plot_accuracy(accuracy):
    fig=plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(8,4)
    plt.bar(np.arange(len(accuracy)), accuracy)
    plt.xticks([])
    plt.xlabel('#Subjects', fontsize=12, fontstyle='oblique')
    plt.ylabel('% Accuracy', fontsize=12, fontstyle='oblique')
    plt.axhline(np.mean(accuracy), color='r', linestyle='--', alpha=.3,
                label=f'{np.mean(accuracy):.2f} ± {np.std(accuracy):.2f}')
    plt.axhline(50, color='k', linestyle='--', alpha=.3, label='chance')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sns.despine()
    fig.savefig('accuracy.png',bbox_inches='tight', pad_inches=0.2, dpi=1200)
    plt.show()
                    
# %%
collector, accuracy = ([] for i in range(0,2))
for count,subject in enumerate(c.subjects_list):
    count += 1
    # load the original dataframe
    data=load_data(subject)
    # get the accuracy
    accuracy.append(cacl_accuracy(data))
    #map to numerical values
    df=tranform_dataframe_for_ANOVA(count,data)
    # get the error rate 
    new=get_the_error_per_unique_distribution(df, count)
    # append to grand collector
    collector.append(new)


data=pd.concat(collector)
data.index.name='subject'
data['error_per_construction']=(data.false_responses/data.trials_per_construction)*1e2
data['error_total']=(data.false_responses/data.total_trials)*1e2
data['error_per_combination']=(data.false_responses/data.trials_per_combination)*1e2

# OUTPUT THE .CSV FILE
fname='data.csv'
data.to_csv(fname)

