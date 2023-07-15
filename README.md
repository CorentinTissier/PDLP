# PDLP
Recently, researchers have proposed a new way to solve very large linear programs: PDLP (https://arxiv.org/abs/2106.04756)  
I have studied the theoretical fundations of PDLP in the PDF Master_Thesis along with some numerical results.  A simplified presentation that takes the form of a poster is given in Poster_PDM.pdf  
  
All the code used to reproduce the PDLP algorithm is contained in the Jupyter Notebook PDLP.ipynb  
mps_loader.py is  a python file that contains the mps loader from the library pysmps (https://github.com/jmaerte/pysmps) with some other lp problems and a way to convert the mps object the numpy arrays c,A,b,G,h,l,u that represents a LP problem.
