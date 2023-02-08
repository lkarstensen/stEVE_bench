# eve_bench
Endovascular Environment: Collection of Endovascular Environments to benchmark autonomous controller

This Repo will be made publicly available and will include selected hardcoded endovascular environments with limited parametrizability. In contrast to this eve_toolbox will be a modular toolbox to quickly prototype new environments where every parameter can be modified. 

We are still searching for good naming. Best option so far:

* **E**ndo**v**ascular **E**nvironment (eve)

The alternatives so far maybe shouldn't be used:

* **Auto**nomous **E**ndovascular **Ro**bo**tic**s Benchmark (AutoERotic Benchmark)
* **A**utonomous **E**ndovascular **Rob**ot**ics** Benchmark (Aerobics Benchmark)

# Environments

| Feature                      | Single realistic Aortic Arch MDP                     | Aortic Arch Generator 2.5D                                                        | Aortic Arch Generator 3D                               | Realistic Aortic Arch POMDP                                              |
| ---------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------ |
| **Feedback**                 | Tracking, (Image?)                                   | Tracking, (Image?)                                                                | Tracking                                               | Tracking                                                                 |
| **Training Geometry(ies)**   | 1 Arch from the Vascular Model Repository            | Aortic Arch Generator, all arteries in one plane. Can be modified with parameters | Aortic Arch Generator. Can be modified with parameters | Several realistic Aortic Arches (e.g. 15/20)                             |
| **Training Device(s)**       | J-Wire (Standard Guidewire)                          | J-Wire (Standard Guidewire)                                                       | J-Wire + J-Catheter                                    | Devices can be choosen from a selection of parametrized standard shapes. |
| **Evaulation Geometry(ies)** | Same as Training                                     | Full Variety from Aortic Arch Generator                                           | Full Variety from Aortic Arch Generator                | Remaining aortic arches (e.g. 5/20)                                      |
| **Evaulation Device(s)**     | Same as Training                                     | Same as Training                                                                  | Same as Training                                       | Same as Training or can be modified with parameters                      |
| **Evaulation Modality**      | Simulation, Testbench Camera, Testbench Fluroroscopy | Simulation, (Testbench Camera?)                                                   | Simulation, (Testbench Camera?)                        | Simulation                                                               |
| **Parameters**               | Difficulty of Targets?,                              | Scaling, Seed, Archtype                                                           | Scaling, Seed, Archtype                                | (Scaling?), Device Parameters, Difficulty                                |


