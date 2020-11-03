# MBRL_contact_rich
Repository for model based hierarchical reinforcement learning for contact rich task. Thesis work at KTH.

# Installing dependencies for gpu support 
Tested on CUDA10.1, Ubuntu18.04, Nvidia GTX1650 MaxQ \
`sudo apt-get install nvidia-driver-440` \
Follow instruction here - https://www.tensorflow.org/install/gpu


# Installing dependencies 
`cd pyPack/gym`\
`pip install -e .`\
`cd ..`\
`cd pyPack/bullet3`\
`pip install -e .`\
`cd ..`\
`cd ..`\
`pip install -r requirements.txt`

# Simulation environments
URDF's location: `pyPack/bullet3/examples/pybullet/gym/pybullet_data`\
Environment scripts location: `pyPack/bullet3/examples/pybullet/gym/pybullet_envs/bullet`\
Simulation enviroments names:\
*2D Block Insertion* - 'BlockInsert2D-v1'\
*2D Block Insertion with change in slot* - 'BlockInsert2Dc-v1'\
*2D Block Insertion with change in initial location*- 'BlockInsert2Dc-v2'\

*2D Block Cornering* - 'BlockSlide2D-v1'\
*2D Block Cornering with friction path* - 'BlockSlide2Dc-v1'\
*2D Block Cornering with change in initial location* - 'BlockSlide2Dc-v2'\


# Running experiments
For base actor critic:\
`python run_bac_policy.py --exp_path='provide experiment path'`

For option-critic:\
`python run_oc_policy.py --exp_path='provide experiment path'`

For model-based option-critic: \
`python run_moc_policy.py --exp_path='provide experiment path'`





