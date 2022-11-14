import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from os import path
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import copy
import os
import numpy as np
import random
import pickle #rick


fps=50
path_project = os.path.abspath(os.path.join("__file__", ".."))
q_object=path_project+"/q_table"
video = os.path.join(path_project, "after_training_theo.mp4")
env = gym.make("ALE/SpaceInvaders-v5",render_mode="rgb_array")


#Setting the hyperparameters
alpha = 0.7 #learning rate                 
discount_factor = 0.8             
true_epsilon= 0.001              
max_epsilon = 0.5
min_epsilon = 0.001         
decay = 1.1         


train_episodes = 250
max_steps = 40*fps #maybe remove q_table file if you reduce this value and pass epsilon to 1
print(max_steps)


def make_frame(t):
    """Returns an image of the frame for time t."""
    # ... create the frame with any library here ...
    return lst[int(t*fps)] # (Height x Width x 3) Numpy array

#STEP 1 - load Q-table or create it

if path.isfile(q_object):
    fileQ=open(q_object, 'rb')
    Q = pickle.load(fileQ)
    fileQ.close()
    while Q.size/env.action_space.n<max_steps:
        Q=np.vstack([Q,np.zeros(env.action_space.n)])
else:
    Q = np.zeros((max_steps, env.action_space.n))

q_object=path_project+"/q_table"
fileQ=open(q_object, 'wb')
#Training the agent

#Creating lists to keep track of reward and epsilon values
total_training_rewards=0
training_rewards = []  
epsilons = []
maxi=0

for episode in range(train_episodes):
    epsilon = true_epsilon
    #Reseting the environment each time as per requirement
    state = env.reset()  
    lst=[]
    print("Epoch:"+str(episode+1)+'/'+str(train_episodes)+" epsi:"+str(epsilon))
    for step in range(max_steps-1):
        
        ### STEP 2: SECOND option for choosing the initial action - exploit     
        #If the random number is larger than epsilon: employing exploitation 
        #and selecting best action 
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(Q[step,:])      
            
        ### STEP 2: FIRST option for choosing the initial action - explore       
        #Otherwise, employing exploration: choosing a random action 
        else:
            action = env.action_space.sample()
    
            
        ### STEPs 3 & 4: performing the action and getting the reward     
        #Taking the action and getting the reward and outcome state
        new_state, reward, done,trunc, info = env.step(action)
        

        ### STEP 5: update the Q-table
        #Updating the Q-table using the Bellman equation
        Q[step, action] = Q[step, action] + alpha * (reward + discount_factor * np.max(Q[step+1, :])) 
        
        total_training_rewards+=reward
        if done: break

    
        #Cutting down on exploration by reducing the epsilon 
        epsilon = epsilon*decay #find good function

     #Adding the total reward and reduced epsilon values
    training_rewards.append(total_training_rewards)
    
    if total_training_rewards > maxi:
        maxi=total_training_rewards
        print("a new reward:"+str(total_training_rewards))
        #save Q_table
        pickle.dump(Q,fileQ)
        
    total_training_rewards=0
    epsilons.append(epsilon)
fileQ.close()   

print("max score:"+str(maxi))
print("Standard deviation: "+str(np.std(training_rewards)))
print ("Mean score: " + str(sum(training_rewards)/train_episodes))


"""Lets see some good gameplay"""

state=env.reset()
lst=[]
done = False
if path.isfile(q_object):
    fileQ=open(q_object, 'rb')
    Q = pickle.load(fileQ)
    fileQ.close()
i=0
check=0
while i<max_steps-1 :
    if done: break
    lst.append(copy.deepcopy(env.render()))
    state, reward, done, _,_ =env.step(np.argmax(Q[i,:])) # take the best action
    check+=reward
    i+=1
    
print("Video_reward:"+str(check))
animation = VideoClip(make_frame, duration=len(lst)//fps) # T-second clip
# export as a video file
animation.write_videofile(video, fps=fps)

env.close()