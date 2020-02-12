# PARC
Parallel Asteroid Rendezvous Code

PARC implements the orbital rendezvous methods from Taylor et al. 2020 (in-prep) in Python 3.7. 

PARC first downloads and imports the MPC Database into the Python environment. 
Next the delta-v calculations for both methods are run in parallel to reduce processing time
An output ascii datafile is generated. 
Prerequisite packages for PARC are: NumPy, VPython, and urllib

PARC takes ~10 minutes to run utilizing 8 threads on a quad core, Hyperthreaded Intel(R) Core(TM) i7-7920HQ CPU at 3.1 GHz. 

Making use of the base Python multiprocessing.Pool() module, the performance of PARC's parallel components scale nearly 
linearly with the number of accessible processing cores.

PARC has been tested on macOS but may encounter problems on non-Unix (Windows) operating systems due to different 
implementations of the multiprocessing module
