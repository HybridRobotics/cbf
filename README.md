# cbf
This repo serves as a toolkit for testing different algorithm for control barrier functions.

**Status:** This repository is still under development, expecting new features/papers and a complete tutorial to explain it. Feel free to raise questions/suggestions through GitHub Issues, if you want to use the current version of this repository. Please watch and star for subscribing further updates which will be related to our latest preprints and published papers.

### Citing
If you find this repository useful in your work, please consider citing following work:

```
@inproceedings{thirugnanam2022safety,
  title={Safety-Critical Control and Planning for Obstacle Avoidance between Polytopes with Control Barrier Functions},
  author={Thirugnanam, Akshay and Zeng, Jun and Sreenath, Koushil},
  booktitle={2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022}
}
```

### Environments
* Create your environment via `conda env create -f environment.yml`. The default conda environment name is `cbf`, and you could also choose that name with your own preferences by editing the .yml file.

### Maze navigation with duality-based obstacle avoidance
This represents the implementation of the following paper:
* A. Thirugnanam, J. Zeng, K. Sreenath. "Safety-Critical Control and Planning for Obstacle Avoidance between Polytopes with Control Barrier Functions." *2022 IEEE International Conference on Robotics and Automation (ICRA)*. [[arXiv]](https://arxiv.org/abs/2109.12313) [[Video]](https://youtu.be/2hKlihdERog)

Run `python models/kinematic_car_test.py`. This simulates maze navigation (two maze setups) with duality-based obstacle avoidance (four robot shapes including rectangle, pentagon, triangle and l-shape) in the discrete-time domain. The animations and snapshots can be found in folder `animations` and `figures`. An example animation video can be generated as follows,

https://user-images.githubusercontent.com/27001847/147999361-faf3557a-3c87-48ab-aa3a-8830b3d565d5.mp4

### Contributors
Jun Zeng, Akshay Thirugnanam.
