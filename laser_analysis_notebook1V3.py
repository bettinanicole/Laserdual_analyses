

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import glob 
import os
import csv 

def update_sham_laser(sham_laser,tag_positions):
    sham_laser_updated=sham_laser+0
    i=0
    for tag in tag_positions:
        tag+=i
        sham_laser_updated[sham_laser_updated>=tag]+=1
        i+=1
    return sham_laser_updated

def extract_allinfo(fileName):                                    
     file_object=pd.read_csv(path+'/'+fileName)                    
     data_temp=pd.DataFrame(data=file_object)                      
     #print data_temp.values                                       
     trialType_s=[]                                                              
     rt_s=[]                                                                     
     accuracy_s=[]   
    
     block_s=[]
     for f in data_temp.values:  
         block_s.append(f[1])
         trialType_s.append(f[4])                                                
         rt_s.append(f[8])                                                       
         accuracy_s.append(f[9])                                                 
     trialType_s=np.array(trialType_s)                                           
     rt_s=np.array(rt_s)                                                         
     accuracy_s=np.array(accuracy_s)  
     block_s=np.array(block_s)
     #print rt_s
                                                                                
     remove_nans=~np.isnan(rt_s)                                                 
     extract_ax=trialType_s=='AX'                                                
     extract_ay=trialType_s=='AY'                                                
     extract_by=trialType_s=='BY'                                                
     extract_bx=trialType_s=='BX'                                                
     extract_ng=trialType_s=='NG' 
     avg_rt_ax=[]                         
     avg_rt_ay=[]                          
     avg_rt_bx=[]
     avg_rt_by=[]
     avg_rt_ng=[] 
    
     avg_rt_ax_true=[]                         
     avg_rt_ay_true=[]                          
     avg_rt_bx_true=[]
     avg_rt_by_true=[]
     avg_rt_ng_true=[]
    
     no_of_nans_ax=[]
     no_of_nans_ay=[]
     no_of_nans_bx=[]
     no_of_nans_by=[]
     no_of_nans_ng=[]
        
     num_true_acc_ax=[]                   
     tot_acc_ax=[]
     num_true_acc_ay=[]                 
     tot_acc_ay=[]                                           

     num_true_acc_by=[]         
     tot_acc_by=[]                                          

     num_true_acc_bx=[]                 
     tot_acc_bx=[]                                        

     num_true_acc_ng=[]                   
     tot_acc_ng=[]
        
     for block in numpy.unique(block_s):
         extract_block=block_s==block
                                                                                 
         avg_rt_ax.append(np.average(rt_s[(extract_ax&remove_nans)&extract_block]))                          
         avg_rt_ay.append(np.average(rt_s[(extract_ay&remove_nans)&extract_block]))                          
         avg_rt_bx.append(np.average(rt_s[(extract_bx&remove_nans)&extract_block]))                          
         avg_rt_by.append(np.average(rt_s[(extract_by&remove_nans)&extract_block]))                          
         avg_rt_ng.append(np.average(rt_s[(extract_ng&remove_nans)&extract_block]))
         
         avg_rt_ax_true.append(np.average(rt_s[(extract_ax&remove_nans)&(extract_block&(accuracy_s==True))]))                          
         avg_rt_ay_true.append(np.average(rt_s[(extract_ay&remove_nans)&(extract_block&(accuracy_s==True))]))                          
         avg_rt_bx_true.append(np.average(rt_s[(extract_bx&remove_nans)&(extract_block&(accuracy_s==True))]))                          
         avg_rt_by_true.append(np.average(rt_s[(extract_by&remove_nans)&(extract_block&(accuracy_s==True))]))                          
         avg_rt_ng_true.append(np.average(rt_s[(extract_ng&remove_nans)&(extract_block&(accuracy_s==True))]))

         #if np.isnan(np.average(rt_s[(extract_ng&remove_nans)&extract_block])):
          #      print rt_s[(extract_ng&remove_nans)&extract_block]
          #      print rt_s[(extract_ng)&extract_block]
            
        

             
         no_of_nans_ax.append(len(rt_s[(extract_ax&(~remove_nans))&extract_block]))                          
         no_of_nans_ay.append(len(rt_s[(extract_ay&(~remove_nans))&extract_block]))

         no_of_nans_bx.append(len(rt_s[(extract_bx&(~remove_nans))&extract_block]))           
         no_of_nans_by.append(len(rt_s[(extract_by&(~remove_nans))&extract_block]))           
         no_of_nans_ng.append(len(rt_s[(extract_ng&(~remove_nans))&extract_block]))           
                                                                   
                                                                                                                                                     
 #    print accuracy_s==True                                                     
         num_true_acc_ax.append(len(rt_s[(extract_ax&(accuracy_s==True))&extract_block]))                   
         tot_acc_ax.append(len(rt_s[extract_ax&extract_block]))                                            

         num_true_acc_ay.append(len(rt_s[(extract_ay&(accuracy_s==True))&extract_block]))                    
         tot_acc_ay.append(len(rt_s[extract_ay&extract_block]))                                            

         num_true_acc_by.append(len(rt_s[(extract_by&(accuracy_s==True))&extract_block]))                  
         tot_acc_by.append(len(rt_s[extract_by&extract_block]))                                            

         num_true_acc_bx.append(len(rt_s[(extract_bx&(accuracy_s==True))&extract_block]))                   
         tot_acc_bx.append(len(rt_s[extract_bx&extract_block]))                                            

         num_true_acc_ng.append(len(rt_s[(extract_ng&(accuracy_s==True))&extract_block]))                    
         tot_acc_ng.append(len(rt_s[extract_ng&extract_block]))                                            
     avg_rt_all=[avg_rt_ax,avg_rt_ay,avg_rt_bx,avg_rt_by,avg_rt_ng]  
     avg_rt_all_true=[avg_rt_ax_true,avg_rt_ay_true,avg_rt_bx_true,avg_rt_by_true,avg_rt_ng_true]              

     no_of_nans_all = [no_of_nans_ax,no_of_nans_ay,no_of_nans_bx,no_of_nans_by, no_of_nans_ng]                                                                       
     num_true_acc_all=[num_true_acc_ax,num_true_acc_ay,num_true_acc_bx,num_true_acc_by,num_true_acc_ng]                                                          
     tot_acc_all = [tot_acc_ax, tot_acc_ay,tot_acc_bx, tot_acc_by,tot_acc_ng] 
        
     return avg_rt_all,no_of_nans_all,num_true_acc_all,tot_acc_all,avg_rt_all_true

'''
data frame look like dis: 
columns = ['subj','trialType','cue','probe','response','rt','accuracy']
'''

#loop through csv files# 

path = '/Users/bettina/Desktop/LaserDual/correctedData/'



#iterate through subjects and separate by condition (laser and sham)

sham_subj = []
laser_subjs = []
sham_laser = []

rt_all = []

rt_all_true=[]

no_of_nans_all = []

num_true_acc_all=[]

tot_acc_all=[]

sham_laser=[]
sham_laser_full=[]



fileNames = os.listdir(path)
for fileName in fileNames[:]:
    #print fileName[3]
    if fileName.endswith(".csv"):
        #print fileName
        #sham_laser.append(fileName[3])
        sham_laser.append(fileName[2:5])
        rt,no_of_nans,num_true_acc,tot_acc,rt_true=extract_allinfo(fileName)
        rt_all.append(rt)
        rt_all_true.append(rt_true)
        no_of_nans_all.append(no_of_nans)
        num_true_acc_all.append(num_true_acc)
        tot_acc_all.append(tot_acc)
            
    #print sham_laser


sham_laser=np.array(sham_laser)
sham_laser=sham_laser.astype(int)
#sham_laser_full=np.array(sham_laser_full)
#sham_laser_full=sham_laser_full.astype(int)
#------------------------Correct for mistakes in id-------------------------------
tag_positions=[]
sham_laser_updated=update_sham_laser(sham_laser,tag_positions)
#---------------------------------------------------------------------------------



extract_even=sham_laser_updated%2==0
extract_odd=~extract_even

rt_all=numpy.array(rt_all)
rt_all_true=numpy.array(rt_all_true)
no_of_nans_all=numpy.array(no_of_nans_all)
num_true_acc_all=numpy.array(num_true_acc_all)
tot_acc_all=numpy.array(tot_acc_all)


##make sure this line works for one subject or try after#





               
                                                                                                                                             
    
#extract_allinfo(fileNames[0])  


# In[60]:


sham_laser


# In[91]:


rt_all[1]


# In[62]:


minimum_accuracy=40.
trial_type_s=[0,1,2,3,4]

block_no_s=[0,1,2,3]
k=0
for sham,linestyle in zip([extract_even,extract_odd],['solid','dashed']):
    average_rt=[]
    average_rt_for_each_block_for_each_type=[]
    average_percentage_nan_for_each_block_for_each_type=[]
    average_percentage_acc_for_each_block_for_each_type=[]
    
    std_rt=[]
    std_rt_for_each_block_for_each_type=[]
    std_percentage_nan_for_each_block_for_each_type=[]
    std_percentage_acc_for_each_block_for_each_type=[]
    
    
    
    rt_all_even_or_odd=rt_all_true[sham]
    no_of_nans_all_even_or_odd=no_of_nans_all[sham]
    num_true_acc_all_even_or_odd=num_true_acc_all[sham]
    tot_acc_all_all_even_or_odd=tot_acc_all[sham]
    for i in trial_type_s: # iterating over trial types
        rt_all_trial_type=rt_all_even_or_odd[:,i]  # extracting each trial type in order [0:- ax, 1:- ay,2:- bx, 3: by, 4: ng ]

        percentage_nan=(no_of_nans_all_even_or_odd.astype(float))[:,i]/tot_acc_all_all_even_or_odd[:,i]*100
        percentage_acc=(num_true_acc_all_even_or_odd.astype(float))[:,i]/tot_acc_all_all_even_or_odd[:,i]*100
        
        
        keep_accurate=((percentage_acc[:,0]>minimum_accuracy)&(percentage_acc[:,1]>minimum_accuracy))&((percentage_acc[:,2]>minimum_accuracy)&(percentage_acc[:,3]>minimum_accuracy))        
        
        
        average_rt_for_each_block=[]
        average_percentage_nan_for_each_block=[]
        average_percentage_acc_for_each_block=[]
        
        std_rt_for_each_block=[]
        std_percentage_nan_for_each_block=[]
        std_percentage_acc_for_each_block=[]
        
    
        
        for j in block_no_s: #iterating over each block

            rt_all_trial_type_block=rt_all_trial_type[:,j]
            
            percentage_acc_block=percentage_acc[:,j]
            percentage_nan_block=percentage_nan[:,j]

            #print rt_all_trial_type_block
            mask=(~np.isnan(rt_all_trial_type_block))&keep_accurate
            average_rt_for_each_block.append(numpy.average(rt_all_trial_type_block[mask]))
            average_percentage_nan_for_each_block.append(numpy.average(percentage_nan_block[mask]))
            average_percentage_acc_for_each_block.append(numpy.average(percentage_acc_block[mask]))
            
            mask=(~np.isnan(rt_all_trial_type_block))&keep_accurate
            std_rt_for_each_block.append(numpy.std(rt_all_trial_type_block[mask])/numpy.sqrt(len(rt_all_trial_type_block[mask])))
            std_percentage_nan_for_each_block.append(numpy.std(percentage_nan_block[mask])/numpy.sqrt(len(percentage_nan_block[mask])))
            std_percentage_acc_for_each_block.append(numpy.std(percentage_acc_block[mask])/numpy.sqrt(len(percentage_acc_block[mask])))
            
   
   
        average_rt_for_each_block_for_each_type.append(average_rt_for_each_block)
        average_percentage_acc_for_each_block_for_each_type.append(average_percentage_acc_for_each_block)
        average_percentage_nan_for_each_block_for_each_type.append(average_percentage_nan_for_each_block)
        
        
        std_rt_for_each_block_for_each_type.append(std_rt_for_each_block)
        std_percentage_acc_for_each_block_for_each_type.append(std_percentage_acc_for_each_block)
        std_percentage_nan_for_each_block_for_each_type.append(std_percentage_nan_for_each_block)

    if (k==0):
        average_rt_for_each_block_for_each_type_even=numpy.transpose(average_rt_for_each_block_for_each_type)
        average_percentage_acc_for_each_block_for_each_type_even=numpy.transpose(average_percentage_acc_for_each_block_for_each_type)
        average_percentage_nan_for_each_block_for_each_type_even=numpy.transpose(average_percentage_nan_for_each_block_for_each_type)
        
        std_rt_for_each_block_for_each_type_even=numpy.transpose(std_rt_for_each_block_for_each_type)
        std_percentage_acc_for_each_block_for_each_type_even=numpy.transpose(std_percentage_acc_for_each_block_for_each_type)
        std_percentage_nan_for_each_block_for_each_type_even=numpy.transpose(std_percentage_nan_for_each_block_for_each_type)
    if (k==1):
        average_rt_for_each_block_for_each_type_odd=numpy.transpose(average_rt_for_each_block_for_each_type)
        average_percentage_acc_for_each_block_for_each_type_odd=numpy.transpose(average_percentage_acc_for_each_block_for_each_type)
        average_percentage_nan_for_each_block_for_each_type_odd=numpy.transpose(average_percentage_nan_for_each_block_for_each_type)
        
        std_rt_for_each_block_for_each_type_odd=numpy.transpose(std_rt_for_each_block_for_each_type)
        std_percentage_acc_for_each_block_for_each_type_odd=numpy.transpose(std_percentage_acc_for_each_block_for_each_type)
        std_percentage_nan_for_each_block_for_each_type_odd=numpy.transpose(std_percentage_nan_for_each_block_for_each_type)
    k+=1
    
    
print average_rt_for_each_block_for_each_type_even

print average_rt_for_each_block_for_each_type_odd


# In[63]:


colors=['red','blue','green','orange']
f,ax=plt.subplots(1,len(block_no_s),figsize=(20,8),sharey=True)

sec_to_millisec=1000.
for i,col in zip(block_no_s,colors)[:]:
    print i
    ax[i].errorbar(trial_type_s,average_rt_for_each_block_for_each_type_even[i]*sec_to_millisec,std_rt_for_each_block_for_each_type_even[i]*sec_to_millisec,marker=' ',color='red',linestyle='solid')
    
    ax[i].errorbar(trial_type_s,average_rt_for_each_block_for_each_type_odd[i]*sec_to_millisec,std_rt_for_each_block_for_each_type_odd[i]*sec_to_millisec,marker=' ',color='blue',linestyle='solid')
    
    ax[i].set_xticks([0,1,2,3,4])
    #ax[i].set_xlabel('Trial type',fontsize=30)
    ax[0].set_ylabel('Reaction time (ms)',fontsize=30)
    ax[i].set_xticklabels(['AX','AY','BX','BY','NG'])
    ax[i].tick_params(labelsize=30)
    
    ax[i].set_xlim(-0.3,4.4)
    

            #

plt.subplots_adjust(wspace=.03)


    


# In[64]:


colors=['red','blue','green','orange']
f,ax=plt.subplots(1,1,figsize=(10,8),sharey=True)

average_rt_over_all_blocks_for_each_type_even=numpy.array([0.]*len(trial_type_s))
average_rt_over_all_blocks_for_each_type_odd=numpy.array([0.]*len(trial_type_s))

std_rt_over_all_blocks_for_each_type_even=numpy.array([0.]*len(trial_type_s))
std_rt_over_all_blocks_for_each_type_odd=numpy.array([0.]*len(trial_type_s))

for i,col in zip(block_no_s,colors)[:]:
    print i
    average_rt_over_all_blocks_for_each_type_even=average_rt_over_all_blocks_for_each_type_even+average_rt_for_each_block_for_each_type_even[i]
    average_rt_over_all_blocks_for_each_type_odd=average_rt_over_all_blocks_for_each_type_odd+average_rt_for_each_block_for_each_type_odd[i]

    std_rt_over_all_blocks_for_each_type_even=std_rt_over_all_blocks_for_each_type_even+std_rt_for_each_block_for_each_type_even[i]
    std_rt_over_all_blocks_for_each_type_odd=std_rt_over_all_blocks_for_each_type_odd+std_rt_for_each_block_for_each_type_odd[i]

    
average_rt_over_all_blocks_for_each_type_even=average_rt_over_all_blocks_for_each_type_even/len(block_no_s)
average_rt_over_all_blocks_for_each_type_odd=average_rt_over_all_blocks_for_each_type_odd/len(block_no_s)

std_rt_over_all_blocks_for_each_type_even=std_rt_over_all_blocks_for_each_type_even/len(block_no_s)
std_rt_over_all_blocks_for_each_type_odd=std_rt_over_all_blocks_for_each_type_odd/len(block_no_s)

ax.errorbar(trial_type_s,average_rt_over_all_blocks_for_each_type_even,std_rt_over_all_blocks_for_each_type_even,marker=' ',color='red',linestyle='solid',label='SHAM')
    
ax.errorbar(trial_type_s,average_rt_over_all_blocks_for_each_type_odd,std_rt_over_all_blocks_for_each_type_odd,marker=' ',color='blue',linestyle='solid', label='LASER')
    
ax.set_xticks([0,1,2,3,4])
    #ax[i].set_xlabel('Trial type',fontsize=30)
ax.set_ylabel('Reaction time (sec)',fontsize=30)
ax.set_xticklabels(['AX','AY','BX','BY','NG'])
ax.set_xlabel('Trial Type', fontsize=30)
plt.title('RT per Trial Type', fontsize=25)

ax.tick_params(labelsize=30)
ax.set_xlim(-0.3,4.4)
ax.legend(loc='upper left',fontsize=20)    
plt.subplots_adjust(wspace=.03)    
    

            #

plt.subplots_adjust(wspace=.03)


    


# In[65]:


print (average_rt_over_all_blocks_for_each_type_even)
print (average_rt_over_all_blocks_for_each_type_odd)
print (std_rt_over_all_blocks_for_each_type_even)
print (std_rt_over_all_blocks_for_each_type_odd)


# In[66]:


colors=['red','blue','green','orange']
f,ax=plt.subplots(1,1,figsize=(10,8),sharey=True)


average_rt_for_each_type_for_each_block_even=numpy.transpose(average_rt_for_each_block_for_each_type_even)
average_rt_for_each_type_for_each_block_odd=numpy.transpose(average_rt_for_each_block_for_each_type_odd)

std_rt_for_each_type_for_each_block_even=numpy.transpose(std_rt_for_each_block_for_each_type_even)
std_rt_for_each_type_for_each_block_odd=numpy.transpose(std_rt_for_each_block_for_each_type_odd)

average_rt_over_all_types_for_each_block_even=numpy.array([0.]*len(block_no_s))
average_rt_over_all_types_for_each_block_odd=numpy.array([0.]*len(block_no_s))

std_rt_over_all_types_for_each_block_even=numpy.array([0.]*len(block_no_s))
std_rt_over_all_types_for_each_block_odd=numpy.array([0.]*len(block_no_s))

for i,col in zip(trial_type_s,colors)[:]:
    print i
    average_rt_over_all_types_for_each_block_even=average_rt_over_all_types_for_each_block_even+average_rt_for_each_type_for_each_block_even[i]
    average_rt_over_all_types_for_each_block_odd=average_rt_over_all_types_for_each_block_odd+average_rt_for_each_type_for_each_block_odd[i]

    std_rt_over_all_types_for_each_block_even=std_rt_over_all_types_for_each_block_even+std_rt_for_each_type_for_each_block_even[i]
    std_rt_over_all_types_for_each_block_odd=std_rt_over_all_types_for_each_block_odd+std_rt_for_each_type_for_each_block_odd[i]

    
average_rt_over_all_types_for_each_block_even=average_rt_over_all_types_for_each_block_even/len(block_no_s)
average_rt_over_all_types_for_each_block_odd=average_rt_over_all_types_for_each_block_odd/len(block_no_s)

std_rt_over_all_types_for_each_block_even=std_rt_over_all_types_for_each_block_even/len(block_no_s)
std_rt_over_all_types_for_each_block_odd=std_rt_over_all_types_for_each_block_odd/len(block_no_s)

ax.errorbar(block_no_s,average_rt_over_all_types_for_each_block_even,std_rt_over_all_types_for_each_block_even,marker=' ',color='red',linestyle='solid',label='SHAM')
    
ax.errorbar(block_no_s,average_rt_over_all_types_for_each_block_odd,std_rt_over_all_types_for_each_block_odd,marker=' ',color='blue',linestyle='solid',label='LASER')
#ax.set_xticks([0,1,2,3,4])
    #ax[i].set_xlabel('Trial type',fontsize=30)
ax.set_ylabel('Reaction time (sec)',fontsize=30)
#ax.set_xticklabels(['AX','AY','BX','BY','NG'])
ax.set_xlabel('Block', fontsize=30)
ax.tick_params(labelsize=30)
#ax.set_xlim(-0.3,4.4)
ax.legend(loc='upper right',fontsize=20)
plt.title('RT per Block', fontsize=25)

            #

plt.subplots_adjust(wspace=.03)


    


# In[68]:



trial_type_s=[0,1,2,3,4]

block_no_s=[0,1,2,3]
k=0
for sham,linestyle in zip([extract_even,extract_odd],['solid','dashed']):
    average_rt=[]
    average_rt_for_each_block_for_each_type=[]
    average_percentage_nan_for_each_block_for_each_type=[]
    average_percentage_acc_for_each_block_for_each_type=[]
    
    std_rt=[]
    std_rt_for_each_block_for_each_type=[]
    std_percentage_nan_for_each_block_for_each_type=[]
    std_percentage_acc_for_each_block_for_each_type=[]
    
    
    
    rt_all_even_or_odd=rt_all_true[sham]
    no_of_nans_all_even_or_odd=no_of_nans_all[sham]
    num_true_acc_all_even_or_odd=num_true_acc_all[sham]
    tot_acc_all_all_even_or_odd=tot_acc_all[sham]
    for i in trial_type_s: # iterating over trial types
        rt_all_trial_type=rt_all_even_or_odd[:,i]  # extracting each trial type in order [0:- ax, 1:- ay,2:- bx, 3: by, 4: ng ]
        #print rt_all_trial_type
        percentage_nan=(no_of_nans_all_even_or_odd.astype(float))[:,i]/tot_acc_all_all_even_or_odd[:,i]*100
        percentage_acc=(num_true_acc_all_even_or_odd.astype(float))[:,i]/tot_acc_all_all_even_or_odd[:,i]*100
        
        
        average_rt_for_each_block=[]
        average_percentage_nan_for_each_block=[]
        average_percentage_acc_for_each_block=[]
        
        std_rt_for_each_block=[]
        std_percentage_nan_for_each_block=[]
        std_percentage_acc_for_each_block=[]
        
    
        
        for j in block_no_s: #iterating over each block

            rt_all_trial_type_block=rt_all_trial_type[:,j]
            
            percentage_acc_block=percentage_acc[:,j]
            percentage_nan_block=percentage_nan[:,j]

            #print rt_all_trial_type_block

            average_rt_for_each_block.append(numpy.average(rt_all_trial_type_block[~np.isnan(rt_all_trial_type_block)]))
            average_percentage_nan_for_each_block.append(numpy.average(percentage_nan_block[~np.isnan(percentage_nan_block)]))
            average_percentage_acc_for_each_block.append(numpy.average(percentage_acc_block[~np.isnan(percentage_acc_block)]))
            
            std_rt_for_each_block.append(numpy.std(rt_all_trial_type_block[~np.isnan(rt_all_trial_type_block)])/numpy.sqrt(len(rt_all_trial_type_block[~np.isnan(rt_all_trial_type_block)])))
            std_percentage_nan_for_each_block.append(numpy.std(percentage_nan_block[~np.isnan(percentage_nan_block)])/numpy.sqrt(len(percentage_nan_block[~np.isnan(percentage_nan_block)])))
            std_percentage_acc_for_each_block.append(numpy.std(percentage_acc_block[~np.isnan(percentage_acc_block)])/numpy.sqrt(len(percentage_acc_block[~np.isnan(percentage_acc_block)])))
            
   
   
        average_rt_for_each_block_for_each_type.append(average_rt_for_each_block)
        average_percentage_acc_for_each_block_for_each_type.append(average_percentage_acc_for_each_block)
        average_percentage_nan_for_each_block_for_each_type.append(average_percentage_nan_for_each_block)
        
        
        std_rt_for_each_block_for_each_type.append(std_rt_for_each_block)
        std_percentage_acc_for_each_block_for_each_type.append(std_percentage_acc_for_each_block)
        std_percentage_nan_for_each_block_for_each_type.append(std_percentage_nan_for_each_block)

    if (k==0):
        average_rt_for_each_block_for_each_type_even=numpy.transpose(average_rt_for_each_block_for_each_type)
        average_percentage_acc_for_each_block_for_each_type_even=numpy.transpose(average_percentage_acc_for_each_block_for_each_type)
        average_percentage_nan_for_each_block_for_each_type_even=numpy.transpose(average_percentage_nan_for_each_block_for_each_type)
        
        std_rt_for_each_block_for_each_type_even=numpy.transpose(std_rt_for_each_block_for_each_type)
        std_percentage_acc_for_each_block_for_each_type_even=numpy.transpose(std_percentage_acc_for_each_block_for_each_type)
        std_percentage_nan_for_each_block_for_each_type_even=numpy.transpose(std_percentage_nan_for_each_block_for_each_type)
    if (k==1):
        average_rt_for_each_block_for_each_type_odd=numpy.transpose(average_rt_for_each_block_for_each_type)
        average_percentage_acc_for_each_block_for_each_type_odd=numpy.transpose(average_percentage_acc_for_each_block_for_each_type)
        average_percentage_nan_for_each_block_for_each_type_odd=numpy.transpose(average_percentage_nan_for_each_block_for_each_type)
        
        std_rt_for_each_block_for_each_type_odd=numpy.transpose(std_rt_for_each_block_for_each_type)
        std_percentage_acc_for_each_block_for_each_type_odd=numpy.transpose(std_percentage_acc_for_each_block_for_each_type)
        std_percentage_nan_for_each_block_for_each_type_odd=numpy.transpose(std_percentage_nan_for_each_block_for_each_type)
    k+=1
    
    
print average_rt_for_each_block_for_each_type_even

print average_rt_for_each_block_for_each_type_odd


# In[69]:


f,ax=plt.subplots(2,len(block_no_s),figsize=(20,16),sharey=True,sharex=True)


for i,col in zip(block_no_s,colors)[:]:
    print i
    ax[0,i].errorbar(trial_type_s,average_percentage_acc_for_each_block_for_each_type_odd[i],std_percentage_acc_for_each_block_for_each_type_odd[i],marker=' ',color='red',linestyle='solid',label='SHAM')
    
    ax[1,i].errorbar(trial_type_s,average_percentage_acc_for_each_block_for_each_type_even[i],std_percentage_acc_for_each_block_for_each_type_even[i],marker=' ',color='blue',linestyle='solid',label='LASER')
    

all_acc_even=[]
all_acc_odd=[]

for k,col in zip([0,1],['red','blue']):
    if (k==0):
        sham=extract_even
    if (k==1):
        sham=extract_odd
    
    tot_acc_all_even_or_odd=tot_acc_all[sham]
    num_true_acc_all_even_or_odd=num_true_acc_all[sham]

    
    student_no_of_blocks_missed=[]
    for tot_acc_each_student,num_true_acc_each_student in zip(tot_acc_all_all_even_or_odd,num_true_acc_all_even_or_odd)[:]:
        
        percentage_acc_each_student=num_true_acc_each_student.astype(float)/tot_acc_each_student*100.
        #print percentage_acc_each_student,num_true_acc_each_student
        temp=numpy.transpose(percentage_acc_each_student)
        
        temp_missed=0
        for i in block_no_s[:]:
            
            if (k==0):
                labl='SHAM'
            if (k==1):
                labl='LASER'
            ax[k,i].errorbar(trial_type_s,temp[i],linestyle=' ',marker='o',color=col)
            if (k==0):
                all_acc_even.append(temp[i])
            if (k==1):
                all_acc_odd.append(temp[i])
                
            
            if (sum(temp[i])==0):
                temp_missed+=1
            
            
            ax[k,i].set_xticks([0,1,2,3,4])
            #ax[i].set_xlabel('Trial type',fontsize=30)
            ax[k,0].set_ylabel('Accuracy (%)',fontsize=30)
            ax[k,i].set_xticklabels(['AX','AY','BX','BY','NG'])
            ax[k,i].tick_params(labelsize=30)
            ax[k,i].set_xlim(-0.3)
        student_no_of_blocks_missed.append(temp_missed)
            #ax[k,i].set_ylim(-1,1)
    if (k==0):
        student_no_of_blocks_missed_even=numpy.array(student_no_of_blocks_missed) 
    if (k==1):
        student_no_of_blocks_missed_odd=numpy.array(student_no_of_blocks_missed) 
plt.subplots_adjust(wspace=0,hspace=0)

import scipy.stats

ax[1,3].legend(fontsize=20)
ax[0,3].legend(fontsize=20)


all_acc_even=numpy.array(all_acc_even)
all_acc_even_flattened=numpy.ndarray.flatten(all_acc_even)

all_acc_odd=numpy.array(all_acc_odd)
all_acc_odd_flattened=numpy.ndarray.flatten(all_acc_odd)

all_acc_even_flattened_filtered=all_acc_even_flattened[(~numpy.isnan(all_acc_even_flattened))]

all_acc_odd_flattened_filtered=all_acc_odd_flattened[(~numpy.isnan(all_acc_odd_flattened))]

scipy.stats.f_oneway(all_acc_even_flattened_filtered,all_acc_odd_flattened_filtered)


# In[70]:


print len(sham_laser)
print len(rt_all)


# In[71]:


f,ax=plt.subplots(2,len(trial_type_s),figsize=(20,16),sharey=True,sharex=True)

for i,col in zip(block_no_s,colors)[:]:
    print i
    ax[0,i].errorbar(trial_type_s,average_rt_for_each_block_for_each_type_even[i],std_rt_for_each_block_for_each_type_even[i],marker=' ',color='red',linestyle='solid')
    
    ax[1,i].errorbar(trial_type_s,average_rt_for_each_block_for_each_type_odd[i],std_rt_for_each_block_for_each_type_odd[i],marker=' ',color='blue',linestyle='solid')
  

all_rts_even=[]
all_rts_odd=[]

for k,col in zip([0,1],['red','blue']):
    if (k==0):
        sham=extract_even
    if (k==1):
        sham=extract_odd
    
    rt_all_even_or_odd=rt_all_true
    
    
    for rt_each_student in rt_all_even_or_odd[:]:
        temp=rt_each_student
        for i in trial_type_s[:]:
            print temp[i]
                    
            ax[k,i].errorbar(block_no_s,temp[i],linestyle=' ',marker='o',color=col)
            
            if (k==0):
                all_rts_even.append(temp[i])
            
            if (k==1):
                all_rts_odd.append(temp[i])
            ax[k,i].set_xticks([0,1,2,3,4])
            #ax[i].set_xlabel('Trial type',fontsize=30)
            ax[k,0].set_ylabel('Reaction time (sec)',fontsize=30)
            ax[k,i].set_xticklabels(['1','2','3','4'])
            ax[k,i].tick_params(labelsize=30)
            ax[k,i].set_xlim(-0.3,4.4)
            
plt.subplots_adjust(wspace=0,hspace=0)


import scipy.stats

all_rts_even=numpy.array(all_rts_even)
all_rts_even_flattened=numpy.ndarray.flatten(all_rts_even)

all_rts_odd=numpy.array(all_rts_odd)
all_rts_odd_flattened=numpy.ndarray.flatten(all_rts_odd)

all_rts_even_flattened_filtered=all_rts_even_flattened[(~numpy.isnan(all_rts_even_flattened))]

all_rts_odd_flattened_filtered=all_rts_odd_flattened[(~numpy.isnan(all_rts_odd_flattened))]

scipy.stats.f_oneway(all_rts_even_flattened_filtered,all_rts_odd_flattened_filtered)


# In[72]:


numpy.sort(sham_laser)


# In[101]:


all_rts_even=[]
all_rts_odd=[]



for cur_student_id in numpy.sort(sham_laser):
    for k,col in zip([0],['red','blue']):
        if (k==0):
            sham=extract_even
        if (k==1):
            sham=extract_odd

        rt_all_even_or_odd=rt_all_true


        for rt_each_student,student_id in zip(rt_all_even_or_odd,sham_laser):
            temp=rt_each_student        
            if (student_id==cur_student_id):
                    #print student_id
                    print cur_student_id,numpy.average(temp[0]),numpy.average(temp[1]),numpy.average(temp[2]),numpy.average(temp[3])
                    #print "----------"
        #df = pd.DataFrame(raw_data, columns = ['cur_student_id,numpy.average(temp[0]),numpy.average(temp[1]),numpy.average(temp[2]),numpy.average(temp[3])
                    #print "----------"
        
                    
        #df=pd.DataFrame(rt_each_student,student_id)


# In[75]:


f,ax=plt.subplots(2,len(block_no_s),figsize=(20,16),sharey=True,sharex=True)

for i,col in zip(block_no_s,colors)[:]:
    print i
    ax[0,i].errorbar(trial_type_s,average_rt_for_each_block_for_each_type_even[i],std_rt_for_each_block_for_each_type_even[i],marker=' ',color='red',linestyle='solid')
    
    ax[1,i].errorbar(trial_type_s,average_rt_for_each_block_for_each_type_odd[i],std_rt_for_each_block_for_each_type_odd[i],marker=' ',color='blue',linestyle='solid')
  

all_rts_even=[]
all_rts_odd=[]

for k,col in zip([0,1],['red','blue']):
    if (k==0):
        sham=extract_even
    if (k==1):
        sham=extract_odd
    
    rt_all_even_or_odd=rt_all_true[sham]
    
    
    for rt_each_student in rt_all_even_or_odd[:]:
        temp=numpy.transpose(rt_each_student)
        for i in block_no_s[:]:
                    
            ax[k,i].errorbar(trial_type_s,temp[i],linestyle=' ',marker='o',color=col)
            
            if (k==0):
                all_rts_even.append(temp[i])
            
            if (k==1):
                all_rts_odd.append(temp[i])
            ax[k,i].set_xticks([0,1,2,3,4])
            #ax[i].set_xlabel('Trial type',fontsize=30)
            ax[k,0].set_ylabel('Reaction time (sec)',fontsize=30)
            ax[k,i].set_xticklabels(['AX','AY','BX','BY','NG'])
            ax[k,i].tick_params(labelsize=30)
            ax[k,i].set_xlim(-0.3,4.4)
            
plt.subplots_adjust(wspace=0,hspace=0)


import scipy.stats

all_rts_even=numpy.array(all_rts_even)
all_rts_even_flattened=numpy.ndarray.flatten(all_rts_even)

all_rts_odd=numpy.array(all_rts_odd)
all_rts_odd_flattened=numpy.ndarray.flatten(all_rts_odd)

all_rts_even_flattened_filtered=all_rts_even_flattened[(~numpy.isnan(all_rts_even_flattened))]

all_rts_odd_flattened_filtered=all_rts_odd_flattened[(~numpy.isnan(all_rts_odd_flattened))]

scipy.stats.f_oneway(all_rts_even_flattened_filtered,all_rts_odd_flattened_filtered)


# In[76]:


colors=['red','blue','green','orange']
f,ax=plt.subplots(1,len(block_no_s),figsize=(20,8),sharey=True)
for i,col in zip(block_no_s,colors)[:]:
    print i
    ax[i].errorbar(trial_type_s,average_percentage_acc_for_each_block_for_each_type_even[i],std_percentage_acc_for_each_block_for_each_type_even[i],marker='o',color='red',linestyle='solid',label='SHAM')
    
    ax[i].errorbar(trial_type_s,average_percentage_acc_for_each_block_for_each_type_odd[i],std_percentage_acc_for_each_block_for_each_type_odd[i],marker='o',color='blue',linestyle='solid',label='LASER')
    
    ax[i].set_xticks([0,1,2,3,4])
    #ax[i].set_xlabel('block',fontsize=30)
    ax[0].set_ylabel('Accuracy (%)',fontsize=30)
    ax[i].set_xticklabels(['AX','AY','BX','BY','NG'])
    ax[i].tick_params(labelsize=30)
    ax[i].set_xlim(-0.3,4.4)

ax[0].legend(loc='upper left',fontsize=20)    
plt.subplots_adjust(wspace=.02)

    


# In[77]:


colors=['red','blue','green','orange']
f,ax=plt.subplots(1,1,figsize=(10,8),sharey=True)

average_percentage_acc_over_all_blocks_for_each_type_even=numpy.array([0.]*len(trial_type_s))
average_percentage_acc_over_all_blocks_for_each_type_odd=numpy.array([0.]*len(trial_type_s))

std_percentage_acc_over_all_blocks_for_each_type_even=numpy.array([0.]*len(trial_type_s))
std_percentage_acc_over_all_blocks_for_each_type_odd=numpy.array([0.]*len(trial_type_s))

for i,col in zip(block_no_s,colors)[:]:
    print i
    average_percentage_acc_over_all_blocks_for_each_type_even=average_percentage_acc_over_all_blocks_for_each_type_even+average_percentage_acc_for_each_block_for_each_type_even[i]
    average_percentage_acc_over_all_blocks_for_each_type_odd=average_percentage_acc_over_all_blocks_for_each_type_odd+average_percentage_acc_for_each_block_for_each_type_odd[i]

    std_percentage_acc_over_all_blocks_for_each_type_even=std_percentage_acc_over_all_blocks_for_each_type_even+std_percentage_acc_for_each_block_for_each_type_even[i]
    std_percentage_acc_over_all_blocks_for_each_type_odd=std_percentage_acc_over_all_blocks_for_each_type_odd+std_percentage_acc_for_each_block_for_each_type_odd[i]

    
average_percentage_acc_over_all_blocks_for_each_type_even=average_percentage_acc_over_all_blocks_for_each_type_even/len(block_no_s)
average_percentage_acc_over_all_blocks_for_each_type_odd=average_percentage_acc_over_all_blocks_for_each_type_odd/len(block_no_s)

std_percentage_acc_over_all_blocks_for_each_type_even=std_percentage_acc_over_all_blocks_for_each_type_even/len(block_no_s)
std_percentage_acc_over_all_blocks_for_each_type_odd=std_percentage_acc_over_all_blocks_for_each_type_odd/len(block_no_s)

ax.errorbar(trial_type_s,average_percentage_acc_over_all_blocks_for_each_type_even,std_percentage_acc_over_all_blocks_for_each_type_even,marker=' ',color='red',linestyle='solid', label = 'SHAM')
    
ax.errorbar(trial_type_s,average_percentage_acc_over_all_blocks_for_each_type_odd,std_percentage_acc_over_all_blocks_for_each_type_odd,marker=' ',color='blue',linestyle='solid', label='LASER')
    
ax.set_xticks([0,1,2,3,4])
    #ax[i].set_xlabel('Trial type',fontsize=30)
ax.set_ylabel('Accuracy',fontsize=30)
ax.set_xticklabels(['AX','AY','BX','BY','NG'])
ax.tick_params(labelsize=30)
ax.set_xlim(-0.3,4.4)
ax.legend(loc='upper right',fontsize=20)
ax.set_xlabel('Trial Type', fontsize=30)
plt.title('Accuracy per Trial Type', fontsize=25)



    

            #

plt.subplots_adjust(wspace=.03)


# In[78]:


print 1-(average_percentage_acc_over_all_blocks_for_each_type_even)/100.
print 1-(average_percentage_acc_over_all_blocks_for_each_type_odd)/100.
print (std_percentage_acc_over_all_blocks_for_each_type_even)/100.
print (std_percentage_acc_over_all_blocks_for_each_type_odd)/100.


# In[79]:


colors=['red','blue','green','orange']
f,ax=plt.subplots(1,1,figsize=(10,8),sharey=True)


average_percentage_acc_for_each_type_for_each_block_even=numpy.transpose(average_percentage_acc_for_each_block_for_each_type_even)
average_percentage_acc_for_each_type_for_each_block_odd=numpy.transpose(average_percentage_acc_for_each_block_for_each_type_odd)

std_percentage_acc_for_each_type_for_each_block_even=numpy.transpose(std_percentage_acc_for_each_block_for_each_type_even)
std_percentage_acc_for_each_type_for_each_block_odd=numpy.transpose(std_percentage_acc_for_each_block_for_each_type_odd)

average_percentage_acc_over_all_types_for_each_block_even=numpy.array([0.]*len(block_no_s))
average_percentage_acc_over_all_types_for_each_block_odd=numpy.array([0.]*len(block_no_s))

std_percentage_acc_over_all_types_for_each_block_even=numpy.array([0.]*len(block_no_s))
std_percentage_acc_over_all_types_for_each_block_odd=numpy.array([0.]*len(block_no_s))

for i,col in zip(trial_type_s,colors)[:]:
    print i
    average_percentage_acc_over_all_types_for_each_block_even=average_percentage_acc_over_all_types_for_each_block_even+average_percentage_acc_for_each_type_for_each_block_even[i]
    average_percentage_acc_over_all_types_for_each_block_odd=average_percentage_acc_over_all_types_for_each_block_odd+average_percentage_acc_for_each_type_for_each_block_odd[i]

    std_percentage_acc_over_all_types_for_each_block_even=std_percentage_acc_over_all_types_for_each_block_even+std_percentage_acc_for_each_type_for_each_block_even[i]
    std_percentage_acc_over_all_types_for_each_block_odd=std_percentage_acc_over_all_types_for_each_block_odd+std_percentage_acc_for_each_type_for_each_block_odd[i]

    
average_percentage_acc_over_all_types_for_each_block_even=average_percentage_acc_over_all_types_for_each_block_even/len(block_no_s)
average_percentage_acc_over_all_types_for_each_block_odd=average_percentage_acc_over_all_types_for_each_block_odd/len(block_no_s)

std_percentage_acc_over_all_types_for_each_block_even=std_percentage_acc_over_all_types_for_each_block_even/len(block_no_s)
std_percentage_acc_over_all_types_for_each_block_odd=std_percentage_acc_over_all_types_for_each_block_odd/len(block_no_s)

ax.errorbar(block_no_s,average_percentage_acc_over_all_types_for_each_block_even,std_percentage_acc_over_all_types_for_each_block_even,marker=' ',color='red',linestyle='solid', label='SHAM')
    
ax.errorbar(block_no_s,average_percentage_acc_over_all_types_for_each_block_odd,std_percentage_acc_over_all_types_for_each_block_odd,marker=' ',color='blue',linestyle='solid', label = 'LASER')
#ax.set_xticks([0,1,2,3,4])
    #ax[i].set_xlabel('Trial type',fontsize=30)
ax.set_ylabel('Accuracy',fontsize=30)
#ax.set_xticklabels(['AX','AY','BX','BY','NG'])
ax.tick_params(labelsize=30)
#ax.set_xlim(-0.3,4.4)
ax.legend(loc='upper right',fontsize=20)
ax.set_xlabel('Block', fontsize=30)
plt.title('Accuracy per Block', fontsize=25)

    

            #

plt.subplots_adjust(wspace=.03)

    




# In[80]:


f,ax=plt.subplots(2,len(block_no_s),figsize=(20,16),sharey=True,sharex=True)


for i,col in zip(block_no_s,colors)[:]:
    print i
    ax[0,i].errorbar(trial_type_s,average_rt_for_each_block_for_each_type_even[i],std_rt_for_each_block_for_each_type_even[i],marker=' ',color='red',linestyle='solid')
    
    ax[1,i].errorbar(trial_type_s,average_rt_for_each_block_for_each_type_odd[i],std_rt_for_each_block_for_each_type_odd[i],marker=' ',color='blue',linestyle='solid')
    

average_percentage_acc_for_each_block_for_each_type_even

for k,col in zip([0,1],['red','blue']):
    if (k==0):
        sham=extract_even
    if (k==1):
        sham=extract_odd
    
    rt_all_even_or_odd=rt_all_true[sham]
    
    for rt_each_student in rt_all_even_or_odd[:]:

        for i in block_no_s[:]:

            temp=numpy.transpose(rt_each_student)
            ax[k,i].errorbar(trial_type_s,temp[i],linestyle=' ',marker='o',color=col)
            
            ax[k,i].set_xticks([0,1,2,3,4])
            #ax[i].set_xlabel('Trial type',fontsize=30)
            ax[k,0].set_ylabel('Reaction time (sec)',fontsize=30)
            ax[k,i].set_xticklabels(['AX','AY','BX','BY','NG'])
            ax[k,i].tick_params(labelsize=30)
            ax[k,i].set_xlim(-0.3,4.4)
            
            
plt.subplots_adjust(wspace=0,hspace=0)


# In[81]:


colors=['red','blue','green','orange']
f,ax=plt.subplots(1,len(block_no_s),figsize=(20,8),sharey=True)
for i,col in zip(block_no_s,colors)[:]:
    print i
    ax[i].errorbar(trial_type_s,average_percentage_nan_for_each_block_for_each_type_even[i],std_percentage_nan_for_each_block_for_each_type_even[i],marker='o',color='red',linestyle='solid',label='SHAM')
    
    ax[i].errorbar(trial_type_s,average_percentage_nan_for_each_block_for_each_type_odd[i],std_percentage_nan_for_each_block_for_each_type_odd[i],marker='o',color='blue',linestyle='solid',label='LASER')
    
    ax[i].set_xticks([0,1,2,3,4])
    #ax[i].set_xlabel('Trial type',fontsize=30)
    ax[0].set_ylabel('Number of NANs',fontsize=30)
    ax[i].set_xticklabels(['AX','AY','BX','BY','NG'])
    ax[i].tick_params(labelsize=30)
    ax[i].set_xlim(-0.3,4.4)
    #ax[i].set_yscale('log')
ax[0].legend(loc='upper left',fontsize=20)
    

plt.subplots_adjust(wspace=.02)

    


# In[82]:


std_percentage_nan_for_each_block_for_each_type_even


# In[83]:


temp=numpy.transpose(average_rt_for_each_block_for_each_type_even)
PBI_RT_even=(temp[1]-temp[2])/(temp[1]+temp[2])
print PBI_RT_even


temp=numpy.transpose(average_rt_for_each_block_for_each_type_odd)
PBI_RT_odd=(temp[1]-temp[2])/(temp[1]+temp[2])
print PBI_RT_odd


# In[84]:


temp=numpy.transpose(average_rt_for_each_block_for_each_type_even)
temp_err=numpy.transpose(std_rt_for_each_block_for_each_type_even)
PBI_RT_even=(temp[1]-temp[2])/(temp[1]+temp[2])
err_numerator=temp_err[1]+temp_err[2]
err_denominator=temp_err[1]+temp_err[2]
relative_err=err_numerator/(temp[1]-temp[2])+err_denominator/(temp[1]+temp[2])
PBI_RT_even_err=PBI_RT_even*relative_err


temp=numpy.transpose(average_rt_for_each_block_for_each_type_odd)
temp_err=numpy.transpose(std_rt_for_each_block_for_each_type_odd)
PBI_RT_odd=(temp[1]-temp[2])/(temp[1]+temp[2])
err_numerator=temp_err[1]+temp_err[2]
err_denominator=temp_err[1]+temp_err[2]
relative_err=err_numerator/(temp[1]-temp[2])+err_denominator/(temp[1]+temp[2])
PBI_RT_odd_err=PBI_RT_odd*relative_err


f,ax=plt.subplots(1,1,figsize=(8,6))

ax.errorbar([1,2,3,4],PBI_RT_even,PBI_RT_even_err,marker='o',ms=15,color='red',label='SHAM')
ax.errorbar([1,2,3,4],PBI_RT_odd,PBI_RT_odd_err,marker='o',ms=15,color='blue',label="LASER")

ax.legend(loc='upper left',fontsize=20)
 

ax.set_xticks([1,2,3,4])
ax.tick_params(labelsize=30)
ax.set_xlabel('Block', fontsize=20)
ax.set_ylabel('RT (sec)', fontsize=20)
plt.title('PBI for RT', fontsize=25)


# In[86]:


f,ax=plt.subplots(1,1,figsize=(8,6))
temp=numpy.transpose(average_percentage_acc_for_each_block_for_each_type_even)
temp_err=numpy.transpose(std_percentage_acc_for_each_block_for_each_type_even)
PBI_percentage_acc_even=(temp[1]-temp[2])/(temp[1]+temp[2])
err_numerator=temp_err[1]+temp_err[2]
err_denominator=temp_err[1]+temp_err[2]
relative_err=err_numerator/(temp[1]-temp[2])+err_denominator/(temp[1]+temp[2])
PBI_percentage_acc_even_err=PBI_percentage_acc_even*relative_err



temp=numpy.transpose(average_percentage_acc_for_each_block_for_each_type_odd)
temp_err=numpy.transpose(std_percentage_acc_for_each_block_for_each_type_odd)
PBI_percentage_acc_odd=(temp[1]-temp[2])/(temp[1]+temp[2])
err_numerator=temp_err[1]+temp_err[2]
err_denominator=temp_err[1]+temp_err[2]
relative_err=err_numerator/(temp[1]-temp[2])+err_denominator/(temp[1]+temp[2])
PBI_percentage_acc_odd_err=PBI_percentage_acc_odd*relative_err




ax.errorbar([1,2,3,4],PBI_percentage_acc_even,PBI_percentage_acc_even_err,marker='o',ms=15, color = 'red', label = 'SHAM')
#print PBI_percentage_acc_even


ax.errorbar([1,2,3,4],PBI_percentage_acc_odd,PBI_percentage_acc_odd_err,marker='o',ms=15,color='blue', label = 'LASER')
#print PBI_percentage_acc_odd
#ax.set_xticks([1,2,3,4])
ax.tick_params(labelsize=30)
ax.set_xlabel('Block', fontsize=20)
ax.set_ylabel('Accuracy', fontsize=20)
plt.title('PBI for Accuracy', fontsize=25)
ax.legend(loc='upper right',fontsize=11)


# In[23]:


temp[1]-temp[2]


# In[ ]:


print PBI_percentage_acc_even
print PBI_percentage_acc_odd

