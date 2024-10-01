__Codes for Particle Positioning and Characterization using Phase Curvature and CNN in Digital Holography__

**Shin-ya Hasegawa and Shota Nakashige**

**Description**

*This repository provides code for:*
1.	Training and estimation using a Convolutional Neural Network (CNN) approach.
2.	Searching for the initial axial position guess of a spherical particle.

**Quickstart**

(1) Python Code for CNN-based Training and Estimation

•	Folder: cnn_code_in_python

•	To run: Execute random_ep10000_4label_5per_noise.py. 

Ensure that random_data_5%noise_csaps_400mic.csv and random_label_5%noise_csaps_400mic.csv are present in the same folder.

(2) MATLAB Code for Initial Axial Position Estimation

•	Folder: axial_position_matlab

•	To run: Execute curvature_calculation.m.

The following files must be in the same folder:

o	fun_calc_curvature.m

o	fun_chebychev_240525.m

o	fun_converge.m

o	fun_maxz_5_190728_rev.m

o	fun_recovery.m

o	holl_100_100.mat

o	holl0_100_100.mat

o	Miguel_2D_unwrapper.cpp

•	Data:

o	holl_100_100.mat: Hologram data recorded using a phase-shifting technique with a particle present.

o	holl0_100_100.mat: Hologram data without a particle.

•	Note: The code utilizes a Mex file (Miguel_2D_unwrapper.cpp). For faster computation after the initial run, you can remove the "mex" call in fun_calc_curvature.m.


Processing Explanation

The processing steps are detailed in our previous paper [1]. To solve the twin image problem, we employ a tilted reference wave for single-shot phase-shifting. This method introduces a real-time phase shift between three adjacent pixels on the camera by adjusting the oblique reference wave [2].

To extract the phase curvature from a particle (Fig. 1):
1.	The reconstructed image is computed using the phase-shifting algorithm and the angular spectrum method [2].
2.	The centroid (xp,yp) of the transversal intensity of the reconstructed particle is determined.
3.	The point of minimal intensity along the z-direction within the centroid is found.
4.	The zero-crossing point of the curvature pattern around this point of minimal intensity is calculated and selected as the zp guess.
5.	Curvatures are computed using the phase φ(xp,yp,zp)  at several designated points, excluding zp.

![Opt_Cont](https://github.com/user-attachments/assets/5d07e9d7-4c68-41d7-9b49-8acc784759f9)
Fig.1 The pipeline of curvature calculation for experimental data. A reconstructed wave is obtained from the angular spectrum method using a hologram [1]. 

References
1.	S. Hasegawa and T. Miaki, "Machine learning techniques for positioning and characterization of particles in digital holography using the whole phase curvature," Opt. Continuum 1, 2561-2576 (2022)]
2.	S. Hasegawa and T. Miaki, "Whole phase curvature-based particle positioning and size determination by digital holography," Appl. Opt. 59, 7201-7210 (2020)
