import numpy as np
import pandas as pd
data = pd.DataFrame(data=pd.read_csv('/candidate_elim.dataset.csv'))  
concepts = np.array(data.iloc[:,0:-1]) 
print('Concepts: \n', concepts)  

target = np.array(data.iloc[:,-1])   
print('\nTarget: \n', target) 
 
def learn(concepts, target):  
    print("\nInitialization of specific_h and general_h: ")      
    
    specific_h = concepts[0].copy()      
    print('\t specific_h:', specific_h)

    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]      
    print('\t general_h:', general_h)

    for i, h in enumerate(concepts):  
        if target[i] == "Yes":  
            for x in range(len(specific_h)):  
                if h[x]!= specific_h[x]:                     
                    specific_h[x] ='?'                      
                    general_h[x][x] ='?'  
        if target[i] == "No":             
            for x in range(len(specific_h)):  
                if h[x]!= specific_h[x]:                     
                    general_h[x][x] = specific_h[x]                 
                else:                     
                    general_h[x][x] = '?'         
        
        print("\n Step ",i+1)
        print('\t specific_h', specific_h)       
        print('\t general_h:', general_h)

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?']]     
    for i in indices:    
        general_h.remove(['?', '?', '?', '?', '?'])  
    return specific_h, general_h

s_final, g_final = learn(concepts, target)

print("\n Final specific_h:", s_final, sep="\n") 
print("\n Final general_h:", g_final, sep="\n")
# Concepts: 
#  [['Sunny' 'Mild' 'High' 'Strong' 'Same']
#  ['Rainy' 'Hot' 'High' 'Normal' 'Same']
#  ['Sunny' 'Mild' 'Normal' 'Strong' 'Change']
#  ['Sunny' 'Hot' 'High' 'Strong' 'Change']
#  ['Sunny' 'Cool' 'Normal' 'Normal' 'Change']
#  ['Overcast' 'Cool' 'Normal' 'Normal' 'Same']
#  ['Rainy' 'Hot' 'Normal' 'Strong' 'Same']]

# Target: 
#  ['Yes' 'No' 'Yes' 'Yes' 'No' 'No' 'Yes']

# Initialization of specific_h and general_h: 
# 	 specific_h: ['Sunny' 'Mild' 'High' 'Strong' 'Same']
# 	 general_h: [['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?']]

#  Step  1
# 	 specific_h ['Sunny' 'Mild' 'High' 'Strong' 'Same']
# 	 general_h: [['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?']]

#  Step  2
# 	 specific_h ['Sunny' 'Mild' 'High' 'Strong' 'Same']
# 	 general_h: [['Sunny', '?', '?', '?', '?'], ['?', 'Mild', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', 'Strong', '?'], ['?', '?', '?', '?', '?']]

#  Step  3
# 	 specific_h ['Sunny' 'Mild' '?' 'Strong' '?']
# 	 general_h: [['Sunny', '?', '?', '?', '?'], ['?', 'Mild', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', 'Strong', '?'], ['?', '?', '?', '?', '?']]

#  Step  4
# 	 specific_h ['Sunny' '?' '?' 'Strong' '?']
# 	 general_h: [['Sunny', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', 'Strong', '?'], ['?', '?', '?', '?', '?']]

#  Step  5
# 	 specific_h ['Sunny' '?' '?' 'Strong' '?']
# 	 general_h: [['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', 'Strong', '?'], ['?', '?', '?', '?', '?']]
# Step  6
# 	 specific_h ['Sunny' '?' '?' 'Strong' '?']
# 	 general_h: [['Sunny', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', 'Strong', '?'], ['?', '?', '?', '?', '?']]

#  Step  7
# 	 specific_h ['?' '?' '?' 'Strong' '?']
# 	 general_h: [['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?'], ['?', '?', '?', 'Strong', '?'], ['?', '?', '?', '?', '?']]

#  Final specific_h:
# ['?' '?' '?' 'Strong' '?']

#  Final general_h:
# [['?', '?', '?', 'Strong', '?']]