#!/usr/bin/env python

from itertools import combinations
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import PandasTools
from rdkit.Chem.SaltRemover import SaltRemover
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *


def df_brics_frag_gen(list_smiles_n):
    # STEP 1 of ARGE function

    # Generation of reference table : Describe all fragments combinations of all molecules depending on brics bonds cuts
    # A number of brics bonds cuts > 1 is used in a combination
    # return : fragments table with smiles_mol, r0, rn, sd, r0_ha
    # smiles_mol: smiles of the molecule used for the fragmentation
    # r0 : core fragment (smiles)
    # rn : all except r0 (smiles)
    # r0_ha : Heavy Atom number of r0
    # sd : standard deviation of Heavy Atom number of all fragments from the same fragmentation

    list_final_r0 = []
    list_final_mol = []
    list_final_rn = []
    list_final_sd = []

    for smiles in list_smiles_n:
        # STEP 1: Determine the maximum number of cuts needed (nb_cuts_max) in a combination to describe all possible fragments

        # In order to describe all fragments combinations of a molecule depending on brics bonds, it is not necessary to consider all brics bonds combinations
        # After a certain number of cuts in a combination, only fragments previously described with a lower number of cuts are obtained
        # The maximum and optimal number of cuts in a combination to consider is defined as the number of one-edge-fragments in the molecule fragmented with all brics cuts
        # The number of edges of a fragment is equivalent to the number of stars in its smiles (nb_star)

        mol = Chem.MolFromSmiles(smiles)
        list_brics_bonds_mol = list(Chem.BRICS.FindBRICSBonds(mol))

        list_brics_bonds_index = []
        for bond in list_brics_bonds_mol:
            list_brics_bonds_index.append(mol.GetBondBetweenAtoms(bond[0][0], bond[0][1]).GetIdx())

        if len(list_brics_bonds_index) != 0:
            fragmented_mol = Chem.FragmentOnBonds(mol, list_brics_bonds_index)
            fragmented_smiles = Chem.MolToSmiles(fragmented_mol)
            list_fragments = fragmented_smiles.split('.')

            nb_star = 0
            for fragment in list_fragments:
                if fragment.count("*") == 1:
                    nb_star = nb_star + 1
            nb_cuts_max = nb_star

            # STEP 2: Fragmentation of molecules with a combination of n brics bonds cuts
            # Iteration with n-cuts combination
            # From 2 cuts combination to nb_cuts_max

            nb_cuts = 2
            while nb_cuts <= nb_cuts_max:

                comb = combinations(list_brics_bonds_index, nb_cuts)

                list_fragmented_smiles = []
                for bonds_combination in list(comb):
                    fragmented_mol = Chem.FragmentOnBonds(mol, bonds_combination)
                    list_fragmented_smiles.append(Chem.MolToSmiles(fragmented_mol, isomericSmiles=False))

                list_list_fragments = []
                for fragmented_smiles in list_fragmented_smiles:
                    list_list_fragments.append(fragmented_smiles.split('.'))

                for list_frag in list_list_fragments:
                    for fragment in list_frag:

                        if fragment.count("*") == nb_cuts:
                            # Only frag with fragment.count("*") == nb_cuts is considered as R0 because others should have been described with a lower iteration

                            list_frag_r0_ha = []
                            for frag in list_frag:
                                list_frag_r0_ha.append(Chem.MolFromSmiles(frag).GetNumHeavyAtoms())
                            list_final_sd.append(np.std(list_frag_r0_ha))

                            # Rn represents all fragments without R0
                            list_rn = list_frag
                            list_rn.remove(fragment)
                            rn = ".".join(list_rn)
                            list_final_rn.append(rn)

                            list_final_r0.append(Chem.MolToSmiles(Chem.MolFromSmiles(fragment)))
                            list_final_mol.append(Chem.MolToSmiles(mol))

                nb_cuts = nb_cuts + 1

    # STEP3: From list to dict, from dict to dataframe
    # duplicates with same r0 and mol are dropped, r0 with r0_ha < 6 are dropped

    dict_frag = {"mol_smiles": list_final_mol, "r0_smiles": list_final_r0, "rn": list_final_rn, "sd": list_final_sd}
    df_brics_frag = pd.DataFrame(dict_frag)

    df_brics_frag = df_brics_frag.drop_duplicates(subset=["mol_smiles", "r0_smiles"], keep="first")

    df_brics_frag["r0_ha"] = df_brics_frag["r0_smiles"].apply(lambda x: Chem.MolFromSmiles(x).GetNumHeavyAtoms())
    df_brics_frag = df_brics_frag.loc[df_brics_frag["r0_ha"] > 5]

    print("STEP 1 succeed: all molecules are fragmented depending on combinations of their BRICS bonds")

    return df_brics_frag


def df_unique_frag_gen(df_brics_frag):
    # STEP 2 of ARGE function: R0 ranking depending on a score

    # From df_brics_frag, qualify all unique r0 with count, sd_mean, score
    # count: occurence of a r0 in the dataset
    # sd_mean : mean of all sd from a unique r0
    # score: mean of log(count) standardized, -(sd_mean standardized) and log(r0_ha) standardized

    # Choice of score parameters:
    # Count: related to R0 occurence in the dataset and to brics bonds occurence (is the R0 frequent in the dataset and synthetically interesting)
    # sd_mean: is R0 size is balanced compared to other substituants
    # R0_HA: the bigger R0 is, the more specific and specific it should be

    df_unique_frag = df_brics_frag.copy()
    df_unique_frag["count"] = df_unique_frag["r0_smiles"].apply(lambda x: df_unique_frag['r0_smiles'].value_counts()[x])
    df_unique_frag["sd_mean"] = df_unique_frag["r0_smiles"].apply(
        lambda x: df_unique_frag["sd"].loc[df_unique_frag["r0_smiles"] == x].mean(axis=0))

    # Keep unique r0
    df_unique_frag.drop_duplicates(subset="r0_smiles", keep="first", inplace=True)

    # Score: - sd_mean_stdz + count_log_stdz + r0_ha_log_stdz) / 3
    df_unique_frag["sd_mean_stdz"] = df_unique_frag["sd_mean"].apply(
        lambda x: (x - df_unique_frag.sd_mean.mean()) / df_unique_frag.sd_mean.std() if df_unique_frag.sd_mean.std() != 0 else 0)

    df_unique_frag["r0_ha_log"] = np.log(df_unique_frag['r0_ha'])
    df_unique_frag["r0_ha_log_stdz"] = df_unique_frag["r0_ha_log"].apply(
        lambda x: (x - df_unique_frag.r0_ha_log.mean()) / df_unique_frag.r0_ha_log.std() if df_unique_frag.r0_ha_log.std() != 0 else 0)

    df_unique_frag["count_log"] = np.log(df_unique_frag['count'])
    df_unique_frag["count_log_stdz"] = df_unique_frag["count_log"].apply(
        lambda x: (x - df_unique_frag.count_log.mean()) / df_unique_frag.count_log.std() if df_unique_frag.count_log.std() != 0 else 0)

    df_unique_frag["Score"] = (
                                          - df_unique_frag.sd_mean_stdz + df_unique_frag.count_log_stdz + df_unique_frag.r0_ha_log_stdz) / 3
    df_unique_frag = df_unique_frag.sort_values(["Score"], ascending=[False])

    df_unique_frag.reset_index(drop=True, inplace=True)

    # Drop residual columns which are related to a R0 of a specific molecule
    del df_unique_frag["mol_smiles"]
    del df_unique_frag["rn"]
    del df_unique_frag["sd"]

    return df_unique_frag


def df_subs_r0_gen(df_unique_frag, list_smiles_n):
    # STEP 3 of ARGE function:
    # best r0 of df_unique_frag is chosen, all molecules with a r0-SubstructMatch are described
    # return table with n*, r0, mol_mol, mol_smiles
    # n* : all r0 substituants depending on their position on r0
    # ex: the substituant with a position of 9* will branch r0 on its 10th atom (ranking from 0)
    # mol_mol: mol object

    # GOAL: describe substituants of all molecules containing best r0

    list_final_dict = []

    best_r0_smiles = df_unique_frag["r0_smiles"][0]
    best_r0_mol = Chem.MolFromSmiles(best_r0_smiles)

    # Smiles with aromatic cycles and edges (*) are cleaned in order to use properly HasSubstructMatch()
    Chem.Kekulize(best_r0_mol, clearAromaticFlags=True)
    best_r0_clean_smiles = Chem.MolToSmiles(best_r0_mol)
    best_r0_clean_smiles = best_r0_clean_smiles.replace("(*)", "")
    best_r0_clean_smiles = best_r0_clean_smiles.replace("*", "")
    best_r0_clean_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(best_r0_clean_smiles))

    patt = Chem.MolFromSmarts(best_r0_clean_smiles)

    for smiles in list_smiles_n:

        if Chem.MolFromSmiles(smiles).HasSubstructMatch(patt):

            # Molecules which match R0 are splited with ReplaceCore
            dict_r0 = {"mol_smiles": smiles, "r0_smiles": best_r0_clean_smiles}

            core = Chem.MolFromSmiles(best_r0_clean_smiles)
            tmp = Chem.ReplaceCore(Chem.MolFromSmiles(smiles), core, labelByIndex=True)

            list_substituants = Chem.MolToSmiles(tmp).split(".")

            for substituant in list_substituants:
                # Substituants are written with "[n*]" to indicate their atomic position on R0 except the 0 position which is indicated only with "*"

                substituant_clean = substituant

                # Clean the process to have columns homogeneity and avoid columns overlaps
                # Ex: position 0 is described as "*" and we want [0*] for homogeneity
                # Ex: columns overlap when there are two substituents on the same position: we want to separate them as [0*] and [0**] for example

                for x in range(0, len(substituant_clean) - 1):
                    if substituant_clean[x] == "*" and substituant_clean[x + 1] != "]":
                        substituant_clean = list(substituant_clean)  # Convert the string to a list

                        substituant_clean[x] = "$"  # Change the character using its index

                        substituant_clean = "".join(substituant_clean)
                substituant_clean = substituant_clean.replace("$", "[0*]")

                for j in range(0, Chem.MolFromSmiles(best_r0_clean_smiles).GetNumHeavyAtoms() + 1):
                    position = "[" + str(j) + "*]"
                    position_clean = str(j) + "*"

                    n = "*"
                    while position_clean in dict_r0:
                        position_clean = position_clean + "*"

                    if position in substituant_clean:
                        substituant_clean = substituant_clean.replace(position, "[" + position_clean + "]")
                        dict_r0[position_clean] = substituant_clean

            list_final_dict.append(dict_r0)

    df_subs_r0 = pd.DataFrame(list_final_dict)

    return df_subs_r0


def r0_clean(df_subs_r0, df_unique_frag, num_ite):
    # STEP 3-bis of ARGE function: convert substituant positions into R1, R2, Rn...
    # Organize best R0 substituants positions and name them as R1, R2, Rn... depending on their occurence (count)

    # Step 1: create a dataframe with substituants name (position) and substituants occurence
    list_columns = ["mol_smiles", "r0_smiles"]
    dict_subs_positions = {"position": [], "occurence": []}

    for pos in df_subs_r0:
        if (pos in list_columns) is False:
            dict_subs_positions["position"].append(pos)
            dict_subs_positions["occurence"].append(df_subs_r0[pos].count())

    df_subs_positions = pd.DataFrame(dict_subs_positions)
    df_subs_positions = df_subs_positions.sort_values(["occurence", "position"], ascending=[False, True])
    df_subs_positions = df_subs_positions.reset_index(drop=True)

    # Organize df_subs_r0 columns depending on substituants occurence
    for pos in df_subs_positions["position"]:
        list_columns.append(pos)
    df_subs_r0 = df_subs_r0[list_columns]

    # Step 2: Attribute a r name (r1,r2 ...) to substituant positions depending on their occurence
    list_r = []
    for nb in range(1, df_subs_positions["position"].count() + 1):
        r = "R" + str(nb)
        list_r.append(r)
    df_subs_positions["R"] = list_r

    for pos in df_subs_positions["position"]:
        df_subs_r0 = df_subs_r0.rename(index=str, columns={
            pos: df_subs_positions["R"].loc[df_subs_positions["position"] == pos].item()})

    # Step 3: Attribute false atoms to each positions and link them to best r0 with AddAtoms() and Addbonds()
    # replace fake atoms with r-names
    df_subs_positions["Atom_index"] = df_subs_positions["position"].apply(lambda x: x.replace("*", ""))

    list_symbol = []

    n = 0
    for pos in df_subs_positions["position"]:
        list_symbol.append(Chem.Atom(89 + n).GetSymbol())
        n = n + 1

    df_subs_positions["sym"] = list_symbol

    r0 = Chem.MolFromSmiles(df_subs_r0["r0_smiles"][0])
    r0_mw = Chem.RWMol(r0)

    n = 0
    for r in df_subs_positions["R"]:
        r0_mw.AddAtom(Chem.Atom(89 + n))
        r0_mw.AddBond(int(df_subs_positions["Atom_index"][n]), int(df_unique_frag["r0_ha"][0] + n),
                      Chem.BondType.SINGLE)
        n = n + 1

    r0_smiles = Chem.MolToSmiles(r0_mw)

    for symbol in df_subs_positions["sym"]:
        r0_smiles = r0_smiles.replace(symbol, df_subs_positions["R"].loc[df_subs_positions["sym"] == symbol].item())

    print(r0_smiles)
    df_subs_r0["r0_smiles"] = df_subs_r0["r0_smiles"].apply(lambda x: r0_smiles)

    # Step 4: A way to classify all molecules with same best R0, rank them with best R1 occurence, then best R2 occurence, etc...

    df_subs_r0 = df_subs_r0.fillna(0)

    for r in df_subs_positions["R"]:
        df_subs_r0[r + "_count"] = df_subs_r0[r].apply(lambda x: df_subs_r0[r].value_counts()[x] if x != 0 else 0)

    list_r_false = [False] * len(list_r)
    list_r_count = []
    for r in list_r:
        list_r_count.append(r + "_count")

    df_subs_r0 = df_subs_r0.sort_values(list_r_count, ascending=list_r_false)
    df_subs_r0 = df_subs_r0.reset_index(drop=True)

    df_subs_positions["[R]"] = df_subs_positions["R"].apply(lambda x: "[" + x + "]")
    df_subs_positions["[position]"] = df_subs_positions["position"].apply(lambda x: "[" + x + "]")

    for x in df_subs_positions["R"]:
        for y in df_subs_positions["[position]"]:
            df_subs_r0[x] = df_subs_r0[x].apply(
                lambda z: str(z).replace(y, df_subs_positions["[R]"].loc[df_subs_positions["[position]"] == y].item()))

        for n in df_subs_r0:
            if n == x:
                a = x + "_smiles"
                df_subs_r0 = df_subs_r0.rename(columns={x: a})

    # Index each rows as IterationNumber_IndexRow
    num_row = 1
    list_index = []
    for x in df_subs_r0["mol_smiles"]:
        index = str(num_ite) + "_" + str(num_row)
        list_index.append(index)
        num_row = num_row + 1
    df_subs_r0["Name"] = list_index

    return df_subs_r0


def df_brics_frag_ite(df_subs_r0, df_brics_frag):
    # Preparation of df_brics_frag for the next iteration, use df_brics_frag without molecules described with the best R0

    for mol in df_subs_r0["mol_smiles"]:
        df_brics_frag = df_brics_frag.loc[df_brics_frag["mol_smiles"] != mol]
    df_brics_frag = df_brics_frag.reset_index(drop=True)

    return df_brics_frag


def list_smiles_n_ite(df_subs_r0, list_smiles_n):
    # Preparation of list_smiles_n for the next iteration, use list_smiles_n without molecules described with the best R0

    list_inter = list_smiles_n
    for mol in df_subs_r0["mol_smiles"]:
        if mol in list_smiles_n:
            list_inter.remove(mol)
    list_smiles_n = list_inter

    return list_smiles_n


def final_undescribed_mol(list_smiles_n):
    print("ok")

    # Handle molecules in list_smiles_n undescribed in df_brics_frag due to brics bonds number of 0 or 1

    dict_residual = {"mol_smiles": [], "r0_smiles": []}
    for smiles in list_smiles_n:
        dict_residual["mol_smiles"].append(smiles)
        dict_residual["r0_smiles"].append(smiles)
        list_smiles_n.remove(smiles)
    df_residual = pd.DataFrame(dict_residual)

    return df_residual


def main_function(button_load_file):
    # main_function:
    #   1: Open a sdf file
    #   2: ARGE function
    #   3: Save sdf-created file

    button_load_file.config(state="disabled")

    # STEP 1: open the chosen sdf file
    root_filename_open = filedialog.askopenfilename(initialdir="/", title="Open file",
                                                    filetypes=(("sdf files", "*.sdf"), ("all files", "*.*")))

    try:

        var.set("ARGE program is running...")
        root.update_idletasks()

        # STEP 2: ARGE function, return final dataframe
        df_final = ARGE_function(root_filename_open)

        print("ARGE function succeed")

        # STEP3: export the dataframe to sdf, save sdf file
        root.filename_save = tk.filedialog.asksaveasfilename(initialdir="/", title="Save file",
                                                             filetypes=(("sdf files", "*.sdf"), ("all files", "*.*")),
                                                             defaultextension="*.*")
        Chem.PandasTools.WriteSDF(df_final, root.filename_save, molColName="mol_mol", properties=list(df_final.columns))

        print("dataframe exported to sdf and saved")
        print("success")

        button_load_file.config(state="normal")
        var.set("Success! Load a SDF file to start the program again...")

    except:
        button_load_file.config(state="normal")
        var.set("Fail... Check preconditions and load a SDF file to start the program again...")

        messagebox.showerror("Error",
                             "An error occured, please be sure that preconditions were respected and try again")

    root.update_idletasks()


def ARGE_function(root_filename_open):
    # STEP 1: from a dataset of molecules, generate all possible fragmentations depending brics bounds
    # STEP 2: determine best R0 depending on a score
    # STEP 3: when there is a match, characterize molecules from the dataset with best R0 and substituants associated
    # STEP 4: Iterate the process with only molecules undescribed with best R0

    # STEP 0: Dataset of molecules, sdf required. List of smiles created.
    suppl = Chem.SDMolSupplier(root_filename_open)

    list_smiles_n = []
    remover = SaltRemover()
    for mol in suppl:
        try:
            res = remover.StripMol(mol)
            list_smiles_n.append(Chem.MolToSmiles(res))
        except:
            print("a line of the SDF has been ignored: "+mol)
    print("STEP 0 succeed: file recognized as sdf, all rows recognized as molecules")

    # STEP 1: df_brics_frag_gen(), return df_brics_frag
    # Generation of all fragments combinations of all molecules depending on brics bonds cuts
    # A number of brics bonds cuts > 1 is used in a combination
    df_brics_frag = df_brics_frag_gen(list_smiles_n)

    # Iterative process, results are compilated in df_final
    dict_0 = {}
    df_final = pd.DataFrame(dict_0)

    num_ite = 1
    while len(list_smiles_n) > 0:

        # STEP 2: R0 ranking depending on a score
        df_unique_frag = df_unique_frag_gen(df_brics_frag)

        # STEP 3: when it is possible, characterize molecules from the dataset with best R0 and substituants associated
        df_subs_r0 = df_subs_r0_gen(df_unique_frag, list_smiles_n)

        # STEP 3-bis: clean the results, attribute R1, R2, Rn labels to substituants
        df_subs_r0 = r0_clean(df_subs_r0, df_unique_frag, num_ite)

        # Concat results in df_final
        df_final = pd.concat([df_final, df_subs_r0], axis=0, sort=False)

        # STEP 4: Prepare df_brics_frag and list_smiles_n of the next iteration
        df_brics_frag = df_brics_frag_ite(df_subs_r0, df_brics_frag)
        list_smiles_n = list_smiles_n_ite(df_subs_r0, list_smiles_n)

        print("num ite: = " + str(num_ite))

        num_ite = num_ite + 1

        if len(df_brics_frag) == 0:
            # Handle molecules in list_smiles_n undescribed in df_brics_frag due to brics bonds number of 0 or 1
            df_final = pd.concat([df_final, final_undescribed_mol(list_smiles_n)], axis=0, sort=False)

    print("Iterative process succeed and residual molecules added")

    df_final["mol_mol"] = df_final["mol_smiles"].apply(lambda x: Chem.MolFromSmiles(x))

    df_final = df_final.fillna(0)
    for x in df_final:
        df_final[x] = df_final[x].apply(lambda n: "" if n == 0 else n)
        df_final[x] = df_final[x].apply(lambda n: "" if n == "0" else n)

    return df_final


def entry_point():
    main()


def main():
    global root, frame, var
    # Display the frame and launch the main_function with button_load_file
    # main_function:
    #   1: Open a sdf file
    #   2: ARGE function
    #   3: Save sdf-created file

    root = tk.Tk()
    frame = tk.Frame(root)
    frame.pack()

    root.title("ARGE")

    var = StringVar()
    var.set('Load a SDF file to start the program...')

    status = Label(root, textvariable=var, bd=1, relief=SUNKEN, anchor=W)
    status.pack(side=tk.BOTTOM, fill=X)

    explanation = "Welcome To The ARGE program !\n\nPlease respect preconditions below:\n\t1. Import a SDF file" \
                  "\n\t2. No empty rows"
    tk.Label(root, justify=tk.LEFT, padx=90, text=explanation).pack(side="left")

    button_quit = tk.Button(frame, text="QUIT", fg="red", command=quit)
    button_quit.pack(side=tk.LEFT)
    button_load_file = tk.Button(frame, text="Load File...", command=lambda: main_function(button_load_file))
    button_load_file.pack(side=tk.LEFT)

    root.mainloop()


if __name__ == '__main__':
    main()
