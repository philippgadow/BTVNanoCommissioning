import awkward as ak
import numba as nb
import numpy as np
import pandas as pd
import os, psutil

BTA_HLT = [
    "PFJet40",
    "PFJet60",
    "PFJet80",
    "PFJet140",
    "PFJet200",
    "PFJet260",
    "PFJet320",
    "PFJet400",
    "PFJet450",
    "PFJet500",
    "PFJet550",
    "AK8PFJet40",
    "AK8PFJet60",
    "AK8PFJet80",
    "AK8PFJet140",
    "AK8PFJet200",
    "AK8PFJet260",
    "AK8PFJet320",
    "AK8PFJet400",
    "AK8PFJet450",
    "AK8PFJet500",
    "AK8PFJet550",
    "PFHT180",
    "PFHT250",
    "PFHT370",
    "PFHT430",
    "PFHT510",
    "PFHT590",
    "PFHT680",
    "PFHT780",
    "PFHT890",
    "PFHT1050",
    "BTagMu_AK4DiJet20_Mu5",
    "BTagMu_AK4DiJet40_Mu5",
    "BTagMu_AK4DiJet70_Mu5",
    "BTagMu_AK4DiJet110_Mu5",
    "BTagMu_AK4DiJet170_Mu5",
    "BTagMu_AK4Jet300_Mu5",
    "BTagMu_AK8DiJet170_Mu5",
    "BTagMu_AK8Jet300_Mu5",
    "BTagMu_AK4DiJet20_Mu5_noalgo",
    "BTagMu_AK4DiJet40_Mu5_noalgo",
    "BTagMu_AK4DiJet70_Mu5_noalgo",
    "BTagMu_AK4DiJet110_Mu5_noalgo",
    "BTagMu_AK4DiJet170_Mu5_noalgo",
    "BTagMu_AK4Jet300_Mu5_noalgo",
    "BTagMu_AK8DiJet170_Mu5_noalgo",
    "BTagMu_AK8Jet300_Mu5_noalgo",
    "BTagMu_AK8Jet170_DoubleMu5_noalgo",
]
# mass table from https://github.com/scikit-hep/particle/blob/master/src/particle/data/particle2022.csv and https://gitlab.cern.ch/lhcb-conddb/DDDB/-/blob/master/param/ParticleTable.txt
df_main = pd.read_csv(
    "src/BTVNanoCommissioning/helpers/particle2022.csv", delimiter=",", skiprows=1
)
df_back = pd.read_csv(
    "src/BTVNanoCommissioning/helpers/ParticleTable.csv", delimiter=",", skiprows=4
)
df_main, df_back = df_main.astype({"ID": int}), df_back.astype({"PDGID": int})
main = dict(zip(df_main.ID, df_main.Mass))
backup = dict(zip(df_back.PDGID, df_back["MASS(GeV)"] * 1000))
hadron_mass_table = {**main, **{k: v for k, v in backup.items() if k not in main}}


def is_from_GSP(GenPart):
    QGP = ak.zeros_like(GenPart.genPartIdxMother)
    QGP = (
        (GenPart.genPartIdxMother >= 0)
        & (GenPart.genPartIdxMother2 == -1)
        & (abs(GenPart.parent.pdgId) == 21)
        & (
            (abs(GenPart.parent.pdgId) == abs(GenPart.pdgId))
            | abs(GenPart.parent.pdgId)
            == 21
        )
        & ~(
            (abs(GenPart.pdgId) == abs(GenPart.parent.pdgId))
            & (GenPart.parent.status >= 21)
            & (GenPart.parent.status <= 29)
        )
    )
    rest_QGP = ak.mask(GenPart, ~QGP)
    restGenPart = rest_QGP.parent
    while ak.any(ak.is_none(restGenPart.parent.pdgId, axis=-1) == False):
        mask_forbad = (
            (restGenPart.genPartIdxMother >= 0)
            & (restGenPart.genPartIdxMother2 == -1)
            & (
                (abs(restGenPart.parent.pdgId) == abs(restGenPart.pdgId))
                | abs(restGenPart.parent.pdgId)
                == 21
            )
            & ~(
                (abs(restGenPart.pdgId) == abs(GenPart.parent.pdgId))
                & (restGenPart.parent.status >= 21)
                & (restGenPart.parent.status <= 29)
            )
        )
        mask_forbad = ak.fill_none(mask_forbad, False)
        QGP = (mask_forbad & ak.fill_none((abs(restGenPart.pdgId == 21)), False)) | QGP

        rest_QGP = ak.mask(restGenPart, (~QGP) & (mask_forbad))
        restGenPart = rest_QGP.parent
        if ak.all(restGenPart.parent.pdgId == True):
            break

    return ak.fill_none(QGP, False)


@nb.njit
def to_bitwise_trigger(pass_trig, builder):
    for it in pass_trig:
        # group by every 32 bits
        builder.begin_list()
        for bitidx in range(len(it) // 32 + 1):
            trig = 0
            start = bitidx * 32
            end = min((bitidx + 1) * 32, len(it))
            for i, b in enumerate(it[start:end]):
                trig += b << i
            builder.integer(trig)
        builder.end_list()

    return builder


@nb.vectorize([nb.float64(nb.int64)], forceobj=True)
def get_hadron_mass(hadron_pids):
    return hadron_mass_table[abs(hadron_pids)]


def cumsum(array):
    layout = array.layout
    layout.content
    scan = ak.Array(
        ak.layout.ListOffsetArray64(
            layout.offsets, ak.layout.NumpyArray(np.cumsum(layout.content))
        )
    )
    cumsum_array = ak.fill_none(scan - ak.firsts(scan) + ak.firsts(array), [], axis=0)
    return cumsum_array
