import numpy as np
import random
import math
from random import choice
import matplotlib.pyplot as plt

#Parameters
D= 5
N= 10
lb = -5
ub = 5
Lb= -5*np.ones(D)
Ub= 5*np.ones(D)
max_iter= 10000
limit = 50

#Rastrigin function
def fun(X):
    funsum = 0
    for i in range(D):
        x = X[:,i]
        funsum += x**2 - 10*np.cos(2*np.pi*x)
    funsum += 10*D
    return funsum

#Styblinski-Tang function, (-5,5)
def fun_(X):
    funsum = 0
    for i in range(D):
        x = X[:,i]
        funsum += x**4-16*(x**2)+5*x
    funsum *= 0.5
    return funsum

#1D to 2D with calculate 'fun' function
def transform1Dto2D_fv(num,D,fun):
    substitute = np.zeros(D)
    toCalculate = np.row_stack((num, substitute))
    testOut = fun(toCalculate)
    result = np.delete(testOut, -1, axis=0)
    return result

#Fitness function
def fitness_machine(num):
    fitness = 1/(1+num) if num>= 0 else 1+abs(num)
    return fitness

#Build probability list
def probability_list(fs, fun):
    out = fun(fs)
    fvList = []
    Probability_list = []
    for count_fitness in out:
        fitValue = fitness_machine(count_fitness)
        fvList.append(fitValue)
    sum_fv = sum(fvList)
    for temp in fvList:
        each_pro = temp/sum_fv
        Probability_list.append(each_pro)
    return Probability_list

#Employed Bee phase
def Employed_update_fs(num_,fs,fitness_list, trial):
    fs_row = fs[num_].tolist()
    num = choice(fs_row)
    cvIndex = fs_row.index(num)
    
    waiting_area = np.delete(fs[:,cvIndex],num_)
    
    challenge_num = choice(waiting_area)
    Xnew = num+(random.uniform(-1,1))*(num-challenge_num)
    if Xnew > ub:
        Xnew = ub
    elif Xnew < lb:
        Xnew = lb
    fs_list = []
    for i in range(D):
        fs_list.append(fs[num_][i])
    fs_list[cvIndex] = Xnew
    fs_element = np.asarray(fs_list)
    
    #Judgment elements
    out_ = transform1Dto2D_fv(fs_element,D,fun)
    new_fitness = fitness_machine(out_)
    
    out_compare = transform1Dto2D_fv(fs[num_],D,fun)
    fitness_compare = fitness_machine(out_compare)

    #Judgment
    if new_fitness < fitness_compare:
        fs[num_][cvIndex] = fs[num_][cvIndex]
        trial[num_] += 1
    else:
        fs[num_][cvIndex] = Xnew
        trial[num_] = 0

    out = fun(fs)
    fitness= []
    for i in range(N):
        fv = fitness_machine(out[i])
        fitness.append(fv)
    return fs, out, fitness, trial, num, Xnew


#On-looker Bee phase
def Onlooker_update_fs_step2(num_,fs_step2):
    #choose vairable Xn
    numOnlooker = choice(fs_step2[num_])
    #choose partner
    fs_step2_row = fs_step2[num_].tolist()
    partnerIndex = fs_step2_row.index(numOnlooker)
    waiting_area_ = np.delete(fs_step2[:,partnerIndex],num_)
    partner = choice(waiting_area_)
    Xnew_ = numOnlooker + (random.uniform(-1,1))*(numOnlooker - partner)
    if Xnew_ > ub:
        Xnew_ = ub
    elif Xnew_ < lb:
        Xnew_ = lb
    fs_list_onlooker = []
    for i in range(D):
        fs_list_onlooker.append(fs_step2[num_][i])
    fs_list_onlooker[partnerIndex] = Xnew_
    fs_element_ = np.asarray(fs_list_onlooker)
    
    #Judgment elements
    test_out_onlooker = transform1Dto2D_fv(fs_element_,D,fun)
    new_fitness_ = fitness_machine(test_out_onlooker)
    
    out_compare = transform1Dto2D_fv(fs_step2[num_],D,fun)
    fitness_compare = fitness_machine(out_compare)

    #Judgment
    if new_fitness_ < fitness_compare:
        fs_step2[num_][partnerIndex] = fs_step2[num_][partnerIndex]
        fv_afe[num_] += 1
    else:
        fs_step2[num_][partnerIndex] = Xnew_
        fv_afe[num_] = 0
    return fs_step2, fv_afe
    
#Generate Food Source
fs=np.zeros((N,D))
for i in range(N):
    for j in range(D):
        fs[i,j] = np.random.uniform(-5,5)

#Count f(x)
out = fun(fs)

#Count Fitness value
fitness_list = []
for i in range(N):
    fv = fitness_machine(out[i])
    fitness_list.append(fv)

#Show the Trial    
trial = np.zeros(N).astype(int)

#Set up the best_volumn
best_list = []
Best_Food_Source_list = []

it = 0
while it <= max_iter:
    #Start the while
    #Start the part of Employed Bee
    for i in range(N):
        after_emoloyed = Employed_update_fs(i,fs,fitness_list, trial)

    #Give new variable names
    fs_1 = after_emoloyed[0]
    fv_afe = after_emoloyed[3]
    
    #Build the probability list
    Probability_list = probability_list(fs_1,fun)

    #Onlooker Bee Phase
    fs_step2 = after_emoloyed[0]

    for i in range(N):
        if random.uniform(0,1) < Probability_list[i]:
            onlooker_out = Onlooker_update_fs_step2(i,fs_step2)
            fs_step2_2 = onlooker_out[0]
            fv_afe = onlooker_out[1]
        else: 
            fs_step2_2 = fs_step2
            fv_afe[i] += 1

    #Scout bee phase
    for j in range(N):
        if fv_afe[j] > limit:
            for i in range(D):
                fs_step2_2[j][i] = random.uniform(-5,5)
                fv_afe[j] = 0
                
    #Store the best answer
    final_out = fun(fs_step2)
    best_ = min(final_out)
    best_list.append(best_)
    
    #Cohesion
    fs = fs_step2_2
    trial = fv_afe  

    index_best = np.where(best_)
    index_best = list(final_out).index(best_)
    Food_Source_best = fs_step2[index_best]
    Best_Food_Source_list.append(Food_Source_best)
    it += 1
    
min_best_num = min(best_list)
#arr = list(best_list).index(max_best_num)
arr = best_list.index(min_best_num)
Best_FS_ = Best_Food_Source_list[arr] 


#print('')
np.set_printoptions(suppress=True)
print('Best F(X): ',min_best_num, 'Best food source: ',Best_FS_)

plt.figure(figsize = (15,8))
plt.xlabel("Iteration",fontsize = 15)
plt.ylabel("Value",fontsize = 15)

plt.plot(best_list,linewidth = 2, label = "History Value", color = 'r')
#plt.plot(every_time_value,linewidth = 2, label = "Best value so far", color = 'b')
plt.legend()
plt.show()