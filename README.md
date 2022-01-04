# cbf
This repo serves as a toolkit for testing different algorithm for control barrier functions.

**Status:** This repository is still under development, expecting new features/papers and a complete tutorial to explain it. Feel free to raise questions/suggestions through GitHub Issues, if you want to use the current version of this repository.

### Citing
If you find this project useful in your work, please consider citing following work:

* A. Thirugnanam, J. Zeng, K. Sreenath. "A Fast Computational Optimization for Control and Trajectory Planning for Obstacle Avoidance between Polytopes." *submitted to 2022 IEEE International Conference on Robotics and Automation (ICRA)*. [[arXiv]](https://arxiv.org/abs/2109.12313) [[Video]](https://youtu.be/wucophROPRY) [[Docs]]()

### Environments
* Create your environment via `conda env create -f environment.yml`. The default conda environment name is `cbf`, and you could also choose that name with your own preferences by editing the .yml file.

### Maze navigation with duality-based obstacle avoidance
Run `python models/kinematic_car_test.py`. This simulates maze navigation (two maze setups) with duality-based obstacle avoidance (four robot shapes including rectangle, pentagon, triangle and l-shape) in the discrete-time domain. The animations and snapshots can be found in folder `animations` and `figures`.

### Contributors
Jun Zeng, Akshay Thirugnanam.