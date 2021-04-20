import random
import csv

print("\n Given training dataset is : \n")

with open('../input/datasetfinds/data-find_s.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        attributes.append (row)
        print(row)
        
print("\nThe total number of training instances are : ",len(attributes))

noOfattr = len(attributes[0])-1

print("\nThe initial hypothesis is : ")
init_hypothesis = ['0']*noOfattr
print(init_hypothesis)

for j in range(0,noOfattr):
        init_hypothesis[j] = attributes[0][j]
        
for i in range(0, len(attributes)):
    if attributes[i][noOfattr] == 'Yes':
        print ("\nInstance ", i+1, "is", attributes[i], " and is Positive Instance")
        for j in range(0,noOfattr):
                if attributes[i][j]!=init_hypothesis[j]:
                    init_hypothesis[j]='?'
                else :
                    init_hypothesis[j]= attributes[i][j] 
                    
        print("The hypothesis for the training instance", i+1, " is: " , init_hypothesis, "\n")
        
     
    if attributes[i][noOfattr] == 'No':
        print ("\nInstance ", i+1, "is", attributes[i], " and is Negative Instance")
        print("The hypothesis for the training instance", i+1, " is: " , init_hypothesis, "\n")
        
        

print("\n The Maximally Specific Hypothesis for a given Training Examples :\n")
print(init_hypothesis)



# Given training dataset is : 

# ['Sunny', 'Mild', 'High', 'Strong', 'Same', 'Yes']
# ['Rainy', 'Hot', 'High', 'Normal', 'Same', 'No']
# ['Sunny', 'Mild', 'Normal', 'Strong', 'Change', 'Yes']
# ['Sunny', 'Hot', 'High', 'Strong', 'Change', 'Yes']
# ['Sunny', 'Cool', 'Normal', 'Normal', 'Change', 'No']
# ['Overcast', 'Cool', 'Normal', 'Normal', 'Same', 'No']
# ['Rainy', 'Hot', 'Normal', 'Strong', 'Same', 'Yes']

# The total number of training instances are :  7

# The initial hypothesis is : 
# ['0', '0', '0', '0', '0']

# Instance  1 is ['Sunny', 'Mild', 'High', 'Strong', 'Same', 'Yes']  and is Positive Instance
# The hypothesis for the training instance 1  is:  ['Sunny', 'Mild', 'High', 'Strong', 'Same'] 


# Instance  2 is ['Rainy', 'Hot', 'High', 'Normal', 'Same', 'No']  and is Negative Instance
# The hypothesis for the training instance 2  is:  ['Sunny', 'Mild', 'High', 'Strong', 'Same'] 


# Instance  3 is ['Sunny', 'Mild', 'Normal', 'Strong', 'Change', 'Yes']  and is Positive Instance
# The hypothesis for the training instance 3  is:  ['Sunny', 'Mild', '?', 'Strong', '?'] 


# Instance  4 is ['Sunny', 'Hot', 'High', 'Strong', 'Change', 'Yes']  and is Positive Instance
# The hypothesis for the training instance 4  is:  ['Sunny', '?', '?', 'Strong', '?'] 


# Instance  5 is ['Sunny', 'Cool', 'Normal', 'Normal', 'Change', 'No']  and is Negative Instance
# The hypothesis for the training instance 5  is:  ['Sunny', '?', '?', 'Strong', '?'] 


# Instance  6 is ['Overcast', 'Cool', 'Normal', 'Normal', 'Same', 'No']  and is Negative Instance
# The hypothesis for the training instance 6  is:  ['Sunny', '?', '?', 'Strong', '?'] 


# Instance  7 is ['Rainy', 'Hot', 'Normal', 'Strong', 'Same', 'Yes']  and is Positive Instance
# The hypothesis for the training instance 7  is:  ['?', '?', '?', 'Strong', '?'] 
# The Maximally Specific Hypothesis for a given Training Examples :

# ['?', '?', '?', 'Strong', '?']