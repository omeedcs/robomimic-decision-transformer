<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


# Decision Transformer for Robomimic


<!-- ABOUT THE PROJECT -->
## About The Project
>

Decision Transformer reformulates offline reinforcement learning as a sequence modeling problem that can be effectively solved with large Transformer models. The most related work to our project is the original Decision Transformer. 

<!-- GETTING STARTED -->
## Getting Started
This project requires a working version Pytorch, a working version of robomimic, and Mujoco.


## Get the Datasets:
Dataset download link: https://drive.google.com/drive/folders/1dHMUOSLUr6AwW3PETn1DQO9CWMklT

```bash
cd ~/robomimic/robomimic/scripts # or wherever robomimic was git cloned ...
python download_datasets.py --tasks sim --dataset_types all --hdf5_types low_dim
```
should save to `robomimic/datasets`


## Dataset Types

### Machine-Generated (MG)
Mixture of suboptimal data from state-of-the-art RL agents

### Proficient-Human (PH) and Multi-Human (MH)
500 total, with 200 proficient human and 300 multi-human.
Demonstrations from teleoperators of varying proficiency

### Our setting: ALL data
More challenging combination of MG, MH, and PH
Weighted towards lower-quality MG data

## Dataset Tasks
Lift: lift the cube

Can: pick up the can and place it in proper spot 

We chose these because they have large amounts of low quality machine generated data, which supports our goal of return conditioning on mixed quality data. 

## Our Decision Transformer Architecutre

![new_arc_decision_transformer](https://user-images.githubusercontent.com/61725820/204408002-9ba41db7-3dd6-454d-a79c-bcfcf1398754.gif)

We input state, actions, and returns-to-go into a causal transformer to get our desired actions. We combine the states actions and return to go into one token. This shortens the sequence length and computational requirements. The original decision transformer uses deterministic policy, we train a multi-modal stochastic policy, which helps to better model continuous actions.

# The Semi-Sparse Reward Function:

During development, we found that robomimic uses sparse rewards due to a binary (success or no success) in the sequence data. We attempted to enable dense rewards in robomimic, but found that the dense reward returned was uncorrelated with dataset quality. 

Through debugging, this led to us manually altering the reward function to add a semi-sparse success bonus that decreased on every time step, giving a wider distribution of target RTGs for the decision transformer than the default binary option of success in robomimic. The max sequence is 500, so if you go past 500 time steps you get nothing!

**The Function:** max(500 - success time, 0)

In future work, we hope that this function sees more iterations of development, and possibly altering the actual dense reward and not the function itself. 

With this change, we altered the training data accordingly, as you can see in the new SequenceDataset.

## Results

### Found that return and past-action conditioning can make robomimic tasks more difficult:
<img width="555" alt="Screenshot 2022-11-28 at 7 16 24 PM" src="https://user-images.githubusercontent.com/61725820/204414506-0e24d601-107d-4ea5-b29a-70d2a4fcef63.png">

### Longer sequence modeling improves action prediction and eases problems caused by multi-modal demonstrations:
<img width="545" alt="Screenshot 2022-11-28 at 7 16 46 PM" src="https://user-images.githubusercontent.com/61725820/204414556-c62a22a3-a45c-4929-9bce-9616279e417f.png">

### Decision Transformer lets us model the whole range of returns, not just the expert:
<img width="471" alt="Screenshot 2022-11-28 at 7 18 41 PM" src="https://user-images.githubusercontent.com/61725820/204414774-def64ce3-6538-4ae9-b2c0-96b12492210d.png">



### Data Tables: 

### Task: Lift
### Type: All

<img width="549" alt="Screenshot 2022-11-28 at 7 20 20 PM" src="https://user-images.githubusercontent.com/61725820/204414959-bd8dc39a-6129-4915-a153-dfcf43dddbae.png">

[Naive BC]: Removing the low-quality data allows for expert performance, as in original robomimic

[DT-1, PH Only]: Removing the low-quality data allows for expert performance, as in original robomimic

[DT-20]: Decision Transformer can (mostly) filter the good demonstrations from the machine-generated noise

### Task: Can
### Type: All

<img width="528" alt="Screenshot 2022-11-28 at 7 22 16 PM" src="https://user-images.githubusercontent.com/61725820/204415220-3637ad38-7dfb-424d-8990-c801f53ca4f6.png">

<img width="532" alt="Screenshot 2022-11-28 at 7 20 42 PM" src="https://user-images.githubusercontent.com/61725820/204415001-e8c753bf-cd8a-4d49-b192-4bb32e84669b.png">

[DT-3]: Action and RTG input sequence makes this task significantly more difficult. But DT is much better than naive BC

[DT-3, DT-10, DT-20, all small]: Smaller Transformer sizes decrease performance in the can task

[DT-3, Gaussian, Large]: Standard Gaussian policies are less capable of modeling multi-modal action distributions than our Gaussian Mixture Model default

<!-- CONTACT -->
## Contact

Alex Chandler - alex.chandler@utexas.edu

Jake Grigsby grigsby@cs.utexas.edu

Omeed Tehrani omeed@cs.utexas.edu
