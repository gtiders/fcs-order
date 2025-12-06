from turtle import pos
from ase.io import read,write
from ase.build import make_supercell



poscar=read("./data/POSCAR")

poscar.repeat((2,2,2)).write("POSCAR_repate",format="vasp")

make_supercell(poscar,[[2,0,0],[0,2,0],[0,0,2]],order="atom-major").write("SPOSCAR",format="vasp")