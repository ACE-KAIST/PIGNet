import copy
import math
import pickle
import random

import numpy as np
import torch
from ase import Atom, Atoms
from ase.io import read
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    ChemicalForceFields,
    rdForceFieldHelpers,
    rdFreeSASA,
    rdmolops,
)
from rdkit.Chem.rdForceFieldHelpers import GetUFFVdWParams
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdmolops import (
    CombineMols,
    GetAdjacencyMatrix,
    GetDistanceMatrix,
    SplitMolByPDBResidues,
)
from rdkit.Chem.TorsionFingerprints import CalculateTorsionAngles, CalculateTorsionLists
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import utils

RDLogger.DisableLog("rdApp.*")
random.seed(0)

interaction_types = [
    "saltbridge",
    "hbonds",
    "pication",
    "pistack",
    "halogen",
    "waterbridge",
    "hydrophobic",
    "metal_complexes",
]


def get_period_group(a):
    PERIODIC_TABLE = """
H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,HE                                           
LI,BE,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,NE                                         
NA,MG,1,1,1,1,1,1,1,1,1,1,AL,SI,P,S,CL,AR                                      
K,CA,SC,TI,V,CR,MN,FE,CO,NI,CU,ZN,GA,GE,AS,SE,BR,KR                            
RB,SR,Y,ZR,NB,MO,TC,RU,RH,PD,AG,CD,IN,SN,SB,TE,I,XE                            
CS,BA,LU,HF,TA,W,RE,OS,IR,PT,AU,HG,TL,PB,BI,PO,AT,RN
    """
    pt = dict()
    for i, per in enumerate(PERIODIC_TABLE.split()):
        for j, ele in enumerate(per.split(",")):
            pt[ele] = (i, j)
    period, group = pt[a.GetSymbol().upper()]
    return one_of_k_encoding(period, [0, 1, 2, 3, 4, 5]) + one_of_k_encoding(
        group, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    )


def get_torsion_energy(m):
    mp = ChemicalForceFields.MMFFGetMoleculeProperties(m)
    if mp is None:
        return 0.0
    ffTerms = ("Bond", "Angle", "StretchBend", "Torsion", "Oop", "VdW", "Ele")
    iTerm = "Torsion"
    for jTerm in ffTerms:
        state = iTerm == jTerm
        setMethod = getattr(mp, "SetMMFF" + jTerm + "Term")
        setMethod(state)
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(m, mp)
    e = ff.CalcEnergy()
    return e


def get_epsilon_sigma(m1, m2, mmff=True):
    if mmff:
        try:
            return get_epsilon_sigma_mmff(m1, m2)
        except:
            return get_epsilon_sigma_uff(m1, m2)
    return get_epsilon_sigma_uff(m1, m2)


def get_epsilon_sigma_uff(m1, m2):
    n1 = m1.GetNumAtoms()
    n2 = m2.GetNumAtoms()
    vdw_epsilon, vdw_sigma = np.zeros((n1, n2)), np.zeros((n1, n2))
    m_combine = CombineMols(m1, m2)
    for i1 in range(n1):
        for i2 in range(n2):
            param = GetUFFVdWParams(m_combine, i1, i1 + i2)
            if param is None:
                continue
            d, e = param
            vdw_epsilon[i1, i2] = e
            vdw_sigma[i1, i2] = d
            # print (i1, i2, e, d)
    return vdw_epsilon, vdw_sigma


def get_epsilon_sigma_mmff(m1, m2):
    n1 = m1.GetNumAtoms()
    n2 = m2.GetNumAtoms()
    vdw_epsilon, vdw_sigma = np.zeros((n1, n2)), np.zeros((n1, n2))
    m_combine = CombineMols(m1, m2)
    mp = ChemicalForceFields.MMFFGetMoleculeProperties(m_combine)
    for i1 in range(n1):
        for i2 in range(n2):
            param = mp.GetMMFFVdWParams(i1, i1 + i2)
            if param is None:
                continue
            d, e, _, _ = param
            vdw_epsilon[i1, i2] = e
            vdw_sigma[i1, i2] = d
            # print (i1, i2, e, d)
    return vdw_epsilon, vdw_sigma


def cal_torsion_energy(m):
    energy = 0
    torsion_list, torsion_list_ring = CalculateTorsionLists(m)
    angles = CalculateTorsionAngles(m, torsion_list, torsion_list_ring)
    for idx, t in enumerate(torsion_list):
        indice, _ = t
        indice, angle = indice[0], angles[idx][0][0]
        v = rdForceFieldHelpers.GetUFFTorsionParams(
            m, indice[0], indice[1], indice[2], indice[3]
        )
        hs = [str(m.GetAtomWithIdx(i).GetHybridization()) for i in indice]
        if set([hs[1], hs[2]]) == set(["SP3", "SP3"]):
            n, pi_zero = 3, math.pi
        elif set([hs[1], hs[2]]) == set(["SP2", "SP3"]):
            n, pi_zero = 6, 0.0
        else:
            continue
        energy += (
            0.5 * v * (1 - math.cos(n * pi_zero) * math.cos(n * angle / 180 * math.pi))
        )
    return energy


def cal_internal_vdw(m):
    retval = 0
    n = m.GetNumAtoms()
    c = m.GetConformers()[0]
    d = np.array(c.GetPositions())
    dm = distance_matrix(d, d)
    adj = GetAdjacencyMatrix(m)
    topological_dm = GetDistanceMatrix(m)
    for i1 in range(n):
        for i2 in range(0, i1):
            param = GetUFFVdWParams(m, i1, i2)
            if param is None:
                continue
            d, e = param
            d = d * 1.0
            if adj[i1, i2] == 1:
                continue
            if topological_dm[i1, i2] < 4:
                continue
            retval += e * ((d / dm[i1, i2]) ** 12 - 2 * ((d / dm[i1, i2]) ** 6))
            # print (i1, i2, e, d)
    return retval


def cal_charge(m):
    try:
        charges = AllChem.CalcEEMcharges(m)
        AllChem.ComputeGasteigerCharges(m)
    except:
        charges = None
    if charges is None:
        charges = [
            float(m.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))
            for i in range(m.GetNumAtoms())
        ]
    else:
        for i in range(m.GetNumAtoms()):
            if charges[i] > 3 or charges[i] < -3:
                charges[i] = float(m.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))
    return charges


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(m, atom_i, i_donor, i_acceptor):
    atom = m.GetAtomWithIdx(atom_i)
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(), ["C", "N", "O", "S", "F", "P", "Cl", "Br", "X"]
        )
        + one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED,
            ],
        )
        + one_of_k_encoding_unk(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3, 4])
        + get_period_group(atom)
        + [atom.GetIsAromatic()]
    )  # (9, 6, 7, 7, 24, 1) --> total 54


def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        H.append(atom_feature(m, i, None, None))
    H = np.array(H)
    # if is_ligand:
    #     H = np.concatenate([H, np.zeros((n,27))], 1)
    # else:
    #     H = np.concatenate([np.zeros((n,27)), H], 1)
    return H


def rotate(molecule, angle, axis, fix_com=False):
    """
    Since edge of each molecules are changing by different orientation,
    this funciton used to make molecules rotation-invariant and enables further
    self-supervised leanring.

    :param molecule: rdkit molecule object
    :param anble: angle to rotate,
                  random value between 0, 360
    :param axis: axis for rotation,
                 vector with three random values between 0, 1
                 (a, b, c) -> each values means x, y, z value of the vector
    :return: rotated molecule
    """
    c = molecule.GetConformers()[0]
    d = np.array(c.GetPositions())
    ori_mean = np.mean(d, 0, keepdims=True)
    if fix_com:
        d = d - ori_mean
    atoms = []
    for i in range(len(d)):
        atoms.append(Atom("C", d[i]))
    atoms = Atoms(atoms)
    atoms.rotate(angle, axis)
    new_d = atoms.get_positions()
    if fix_com:
        new_d += ori_mean
    for i in range(molecule.GetNumAtoms()):
        c.SetAtomPosition(i, new_d[i])
    return molecule


def dm_vector(d1, d2):
    """
    Get distances for every atoms in molecule1(ligand), molecule2(protein).
    # of atoms in molecule1, molecule2 = n1, n2 / shape of d1, d2 = [n1, 3], [n2, 3]
    repeat these vectors, make vector shape of [n1, n2, 3] subtract two vectors
    Square value of resulting vector means distances between every atoms in two molecules
    :param d1: position vector of atoms in molecule1, shape: [n1, 3]
    :param d2: position vector of atoms in molecule2, shape: [n2, 3]
    :return: subtraction between enlarged vectors from d1, d2, shape : [n1, n2, 3]
    """
    n1 = len(d1)
    n2 = len(d2)
    d1 = np.repeat(np.expand_dims(d1, 1), n2, 1)
    d2 = np.repeat(np.expand_dims(d2, 0), n1, 0)
    return d1 - d2


def extract_valid_amino_acid(m, amino_acids):
    """
    Divide molecule into PDB residues and only select the residues
    belong to amino acids. Then, combine the all of the residues
    to make new molecule object. This is not 'real' molecule, just
    the information of the molecule with several residues
    :param m: rdkit molecule object
    :param amino_acids: lists of amino acids, total 22 exist
    :return: one molecule information that all of amino acid residues are combined
    """
    ms = SplitMolByPDBResidues(m)
    valid_ms = [ms[k] for k in ms.keys()]
    # valid_ms = [ms[k] for k in ms.keys() if k in amino_acids]
    ret_m = None
    for i in range(len(valid_ms)):
        if i == 0:
            ret_m = valid_ms[0]
        else:
            ret_m = CombineMols(ret_m, valid_ms[i])
    return ret_m


def position_to_index(positions, target_position):
    indice = np.where(np.all(positions == target_position, axis=1))[0]
    diff = positions - np.expand_dims(np.array(target_position), 0)
    diff = np.sum(np.power(diff, 2), -1)
    indice = np.where(diff < 1e-6)[0]
    return indice.tolist()


def get_interaction_matrix(d1, d2, interaction_data):
    n1, n2 = len(d1), len(d2)

    A = np.zeros((len(interaction_types), n1, n2))
    for i_type, k in enumerate(interaction_types):
        for ps in interaction_data[k]:
            p1, p2 = ps
            i1 = position_to_index(d1, p1)
            i2 = position_to_index(d2, p2)
            if len(i1) == 0:
                i1 = position_to_index(d1, p2)
                i2 = position_to_index(d2, p1)
            if len(i1) == 0 or len(i2) == 0:
                pass
            else:
                i1, i2 = i1[0], i2[0]
                A[i_type, i1, i2] = 1
    return A


def classifyAtoms(mol, polar_atoms=[7, 8, 15, 16]):
    # Taken from https://github.com/mittinatten/freesasa/blob/master/src/classifier.c
    symbol_radius = {
        "H": 1.10,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "P": 1.80,
        "S": 1.80,
        "SE": 1.90,
        "FE": 2.05,
        "F": 1.47,
        "CL": 1.75,
        "BR": 1.83,
        "I": 1.98,
        "LI": 1.81,
        "BE": 1.53,
        "B": 1.92,
        "NA": 2.27,
        "MG": 1.74,
        "AL": 1.84,
        "SI": 2.10,
        "K": 2.75,
        "CA": 2.31,
        "GA": 1.87,
        "GE": 2.11,
        "AS": 1.85,
        "RB": 3.03,
        "SR": 2.49,
        "IN": 1.93,
        "SN": 2.17,
        "SB": 2.06,
        "TE": 2.06,
        "MN": 2.05,
    }

    radii = []
    for atom in mol.GetAtoms():
        atom.SetProp("SASAClassName", "Apolar")  # mark everything as apolar to start
        if (
            atom.GetAtomicNum() in polar_atoms
        ):  # identify polar atoms and change their marking
            atom.SetProp("SASAClassName", "Polar")  # mark as polar
        elif atom.GetAtomicNum() == 1:
            if atom.GetBonds()[0].GetOtherAtom(atom).GetAtomicNum() in polar_atoms:
                atom.SetProp("SASAClassName", "Polar")  # mark as polar
        radii.append(symbol_radius[atom.GetSymbol().upper()])
    return radii


def cal_sasa(m):
    radii = rdFreeSASA.classifyAtoms(m)
    radii = classifyAtoms(m)
    # radii = rdFreeSASA.classifyAtoms(m1)
    sasa = rdFreeSASA.CalcSASA(m, radii, query=rdFreeSASA.MakeFreeSasaAPolarAtomQuery())
    return sasa


def get_vdw_radius(a):
    metal_symbols = ["Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu"]
    atomic_number = a.GetAtomicNum()
    atomic_number_to_radius = {
        6: 1.90,
        7: 1.8,
        8: 1.7,
        16: 2.0,
        15: 2.1,
        9: 1.5,
        17: 1.8,
        35: 2.0,
        53: 2.2,
        30: 1.2,
        25: 1.2,
        26: 1.2,
        27: 1.2,
        12: 1.2,
        28: 1.2,
        20: 1.2,
        29: 1.2,
    }
    if atomic_number in atomic_number_to_radius.keys():
        return atomic_number_to_radius[atomic_number]
    return Chem.GetPeriodicTable().GetRvdw(atomic_number)


def cal_uff(m):
    ffu = AllChem.UFFGetMoleculeForceField(m)
    e = ffu.CalcEnergy()
    return e


def get_hydrophobic_atom(m):
    n = m.GetNumAtoms()
    retval = np.zeros((n,))
    for i in range(n):
        a = m.GetAtomWithIdx(i)
        s = a.GetSymbol()
        if s.upper() in ["F", "CL", "BR", "I"]:
            retval[i] = 1
        elif s.upper() in ["C"]:
            n_a = [x.GetSymbol() for x in a.GetNeighbors()]
            diff = list(set(n_a) - set(["C"]))
            if len(diff) == 0:
                retval[i] = 1
        else:
            continue
    return retval


def get_A_hydrophobic(m1, m2):
    indice1 = get_hydrophobic_atom(m1)
    indice2 = get_hydrophobic_atom(m2)
    return np.outer(indice1, indice2)


def get_hbond_donor_indice(m):
    """
    indice = m.GetSubstructMatches(HDonorSmarts)
    if len(indice)==0: return np.array([])
    indice = np.array([i for i in indice])[:,0]

    return indice
    """
    # smarts = ['[!$([#6,H0,-,-2,-3])]', '[!H0;#7,#8,#9]']
    smarts = ["[!#6;!H0]"]
    indice = []
    for s in smarts:
        s = Chem.MolFromSmarts(s)
        indice += [i[0] for i in m.GetSubstructMatches(s)]
    indice = np.array(indice)
    return indice


def get_hbond_acceptor_indice(m):
    # smarts = ['[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
    #          '[#6,#7;R0]=[#8]']
    smarts = ["[$([!#6;+0]);!$([F,Cl,Br,I]);!$([o,s,nX3]);!$([Nv5,Pv5,Sv4,Sv6])]"]
    indice = []
    for s in smarts:
        s = Chem.MolFromSmarts(s)
        indice += [i[0] for i in m.GetSubstructMatches(s)]
    indice = np.array(indice)
    return indice


def get_A_hbond(m1, m2):
    h_acc_indice1 = get_hbond_acceptor_indice(m1)
    h_acc_indice2 = get_hbond_acceptor_indice(m2)
    h_donor_indice1 = get_hbond_donor_indice(m1)
    h_donor_indice2 = get_hbond_donor_indice(m2)
    A = np.zeros((m1.GetNumAtoms(), m2.GetNumAtoms()))

    for i in h_acc_indice1:
        for j in h_donor_indice2:
            A[i, j] = 1
    for i in h_donor_indice1:
        for j in h_acc_indice2:
            A[i, j] = 1

    return A


def get_A_metal_complexes(m1, m2):
    h_acc_indice1 = get_hbond_acceptor_indice(m1)
    h_acc_indice2 = get_hbond_acceptor_indice(m2)
    metal_symbols = ["Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu"]
    metal_indice1 = np.array(
        [
            i
            for i in range(m1.GetNumAtoms())
            if m1.GetAtomWithIdx(i).GetSymbol() in metal_symbols
        ]
    )
    metal_indice2 = np.array(
        [
            i
            for i in range(m2.GetNumAtoms())
            if m2.GetAtomWithIdx(i).GetSymbol() in metal_symbols
        ]
    )
    A = np.zeros((m1.GetNumAtoms(), m2.GetNumAtoms()))

    for i in h_acc_indice1:
        for j in metal_indice2:
            A[i, j] = 1
    for i in metal_indice1:
        for j in h_acc_indice2:
            A[i, j] = 1
    return A


def mol_to_feature(m1, m1_uff, m2, interaction_data, pos_noise_std):
    # Remove hydrogens
    m1 = Chem.RemoveHs(m1)
    m2 = Chem.RemoveHs(m2)

    # extract valid amino acids
    # m2 = extract_valid_amino_acid(m2, self.amino_acids)

    # random rotation
    angle = np.random.uniform(0, 360, 1)[0]
    axis = np.random.uniform(-1, 1, 3)
    # m1 = rotate(m1, angle, axis, False)
    # m2 = rotate(m2, angle, axis, False)

    angle = np.random.uniform(0, 360, 1)[0]
    axis = np.random.uniform(-1, 1, 3)
    m1_rot = rotate(copy.deepcopy(m1), angle, axis, True)

    # prepare ligand
    n1 = m1.GetNumAtoms()
    d1 = np.array(m1.GetConformers()[0].GetPositions())
    d1 += np.random.normal(0.0, pos_noise_std, d1.shape)
    d1_rot = np.array(m1_rot.GetConformers()[0].GetPositions())
    adj1 = GetAdjacencyMatrix(m1) + np.eye(n1)
    h1 = get_atom_feature(m1, True)

    # prepare protein
    n2 = m2.GetNumAtoms()
    c2 = m2.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    d2 += np.random.normal(0.0, pos_noise_std, d2.shape)
    adj2 = GetAdjacencyMatrix(m2) + np.eye(n2)
    h2 = get_atom_feature(m2, True)

    # prepare distance vector
    dmv = dm_vector(d1, d2)
    dmv_rot = dm_vector(d1_rot, d2)

    # get interaction matrix
    # A_int = get_interaction_matrix(d1, d2, interaction_data)
    A_int = np.zeros((len(interaction_types), m1.GetNumAtoms(), m2.GetNumAtoms()))
    A_int[-2] = get_A_hydrophobic(m1, m2)
    A_int[1] = get_A_hbond(m1, m2)
    A_int[-1] = get_A_metal_complexes(m1, m2)

    # cal sasa
    # sasa = cal_sasa(m1)
    # dsasa = sasa-cal_sasa(m1_uff)
    sasa = 0
    dsasa = 0

    # count rotatable bonds
    rotor = CalcNumRotatableBonds(m1)

    # charge
    # charge1 = cal_charge(m1)
    # charge2 = cal_charge(m2)
    charge1 = np.zeros((n1,))
    charge2 = np.zeros((n2,))

    """
    mp1 = AllChem.MMFFGetMoleculeProperties(m1)
    mp2 = AllChem.MMFFGetMoleculeProperties(m2)
    charge1 = [mp1.GetMMFFPartialCharge(i) for i in range(m1.GetNumAtoms())]
    charge2 = [mp2.GetMMFFPartialCharge(i) for i in range(m2.GetNumAtoms())]
    """

    # partial charge calculated by gasteiger
    charge1 = np.array(charge1)
    charge2 = np.array(charge2)

    # There is nan for some cases.
    charge1 = np.nan_to_num(charge1, nan=0, neginf=0, posinf=0)
    charge2 = np.nan_to_num(charge2, nan=0, neginf=0, posinf=0)

    # valid
    valid1 = np.ones((n1,))
    valid2 = np.ones((n2,))

    # no metal
    metal_symbols = ["Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu"]
    no_metal1 = np.array(
        [1 if a.GetSymbol() not in metal_symbols else 0 for a in m1.GetAtoms()]
    )
    no_metal2 = np.array(
        [1 if a.GetSymbol() not in metal_symbols else 0 for a in m2.GetAtoms()]
    )
    # vdw radius
    vdw_radius1 = np.array([get_vdw_radius(a) for a in m1.GetAtoms()])
    vdw_radius2 = np.array([get_vdw_radius(a) for a in m2.GetAtoms()])

    vdw_epsilon, vdw_sigma = get_epsilon_sigma(m1, m2, False)

    # uff energy difference
    # delta_uff = cal_uff(m1)-cal_uff(m1_uff)
    # delta_uff = get_torsion_energy(m1) - get_torsion_energy(m1_uff)
    # delta_uff = cal_torsion_energy(m1)+cal_internal_vdw(m1)
    delta_uff = 0.0
    sample = {
        "h1": h1,
        "adj1": adj1,
        "h2": h2,
        "adj2": adj2,
        "A_int": A_int,
        "dmv": dmv,
        "dmv_rot": dmv_rot,
        "pos1": d1,
        "pos2": d2,
        "sasa": sasa,
        "dsasa": dsasa,
        "rotor": rotor,
        "charge1": charge1,
        "charge2": charge2,
        "vdw_radius1": vdw_radius1,
        "vdw_radius2": vdw_radius2,
        "vdw_epsilon": vdw_epsilon,
        "vdw_sigma": vdw_sigma,
        "delta_uff": delta_uff,
        "valid1": valid1,
        "valid2": valid2,
        "no_metal1": no_metal1,
        "no_metal2": no_metal2,
    }
    return sample


class MolDataset(Dataset):
    def __init__(self, keys, data_dir, id_to_y, random_rotation=0.0, pos_noise_std=0.0):
        self.keys = keys
        self.data_dir = data_dir
        self.id_to_y = id_to_y
        self.random_rotation = random_rotation
        self.amino_acids = [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "ASX",
            "CYS",
            "GLU",
            "GLN",
            "GLX",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ]
        self.pos_noise_std = pos_noise_std

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with open(self.data_dir + "/" + key, "rb") as f:
            data = pickle.load(f)
        if len(data) == 4:
            m1, m1_uff, m2, interaction_data = data
        elif len(data) == 2:
            m1, m2 = data
            m1_uff = 0
            interaction_data = None

        try:
            sample = mol_to_feature(
                m1, m1_uff, m2, interaction_data, self.pos_noise_std
            )
        except Exception as e:
            print(key)
            print(e)
            exit()
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key
        return sample


class DTISampler(Sampler):
    """
    Torch Sampler object that used in Data Loader.
    This simply changes the __iter__ part of the dataset class.
    In this case, we have weight parameter for each data which means importance,
    and sampling will be done by choosing each elements proportionally to this
    weight value.
    Total data size is len(weights) and sampler choose only num_samples number of
    data among total data.
    """

    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights) / np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        retval = np.random.choice(
            len(self.weights),
            self.num_samples,
            replace=self.replacement,
            p=self.weights,
        )
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples


def check_dimension(tensors):
    size = []
    for tensor in tensors:
        if isinstance(tensor, np.ndarray):
            size.append(tensor.shape)
        else:
            size.append(0)
    size = np.asarray(size)

    return np.max(size, 0)


def collate_tensor(tensor, max_tensor, batch_idx):
    if isinstance(tensor, np.ndarray):
        dims = tensor.shape
        max_dims = max_tensor.shape
        slice_list = tuple([slice(0, dim) for dim in dims])
        slice_list = [slice(batch_idx, batch_idx + 1), *slice_list]
        # max_tensor[slice_list] = tensor
        max_tensor[tuple(slice_list)] = tensor
    elif isinstance(tensor, str):
        max_tensor[batch_idx] = tensor
    else:
        max_tensor[batch_idx] = tensor

    return max_tensor


def tensor_collate_fn(batch):
    # batch_items = [it for e in batch for it in e.items() if 'key' != it[0]]
    batch_items = [it for e in batch for it in e.items()]
    dim_dict = dict()
    total_key, total_value = list(zip(*batch_items))
    batch_size = len(batch)
    n_element = int(len(batch_items) / batch_size)
    total_key = total_key[0:n_element]
    for i, k in enumerate(total_key):
        value_list = [v for j, v in enumerate(total_value) if j % n_element == i]
        if isinstance(value_list[0], np.ndarray):
            dim_dict[k] = np.zeros(np.array([batch_size, *check_dimension(value_list)]))
        elif isinstance(value_list[0], str):
            dim_dict[k] = ["" for _ in range(batch_size)]
        else:
            dim_dict[k] = np.zeros((batch_size,))

    ret_dict = {}
    for j in range(batch_size):
        if batch[j] == None:
            continue
        keys = []
        for key, value in dim_dict.items():
            value = collate_tensor(batch[j][key], value, j)
            if not isinstance(value, list):
                value = torch.from_numpy(value).float()
            ret_dict[key] = value

    return ret_dict
