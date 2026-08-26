#!/usr/bin/env python
"""
Full Workflow Example

This workflow:
  - Loads NanoAOD events (using NanoAODSchema)
  - Applies object selections for electrons, muons and jets
  - Forms dilepton pairs and assigns the ttbar channel
  - Computes vectorised kinematic quantities (mll, deltaR, ptrel, etc.)
  - Computes event weights (generator, pileup, trigger, lepton SF, etc.)
  - Fills Coffea histograms
  - Builds two “tree” dictionaries with branch arrays matching the branches
    of your C++ TTbarEventAnalysis: "kin" and "ftm".

All operations are performed fully vectorised using Awkward Arrays.
"""

import os
import awkward as ak
import numpy as np

from coffea import processor
from coffea.nanoevents import NanoAODSchema
from os.path import join

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
    reweighting,
)

# user helper function
from BTVNanoCommissioning.helpers.func import update, dump_lumi
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.helpers.definitions import get_discriminators

## load histograms & selctions for this workflow
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    histo_writter,
)
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    mu_idiso,
    ele_cuttightid,
)


# ----------------------------
# Helper functions
# ----------------------------
def compute_pt_rel(lepton, jet):
    """
    Compute the relative transverse momentum (pt_rel) of the lepton
    with respect to the jet direction using (pt, eta, phi, mass).

    Parameters:
      lepton: a vector or Awkward array of lepton 4-vectors with fields
              "pt", "eta", "phi", "mass", having the same shape as `jet`.
      jet:    a vector or Awkward array of jet 4-vectors with fields
              "pt", "eta", "phi", "mass", having the same shape as `lepton`.

    Returns:
      An array (e.g. an Awkward array or NumPy array) of pt_rel values.
    """

    # Convert lepton kinematics from (pt, eta, phi) to cartesian components.
    lepton_x = lepton.pt * np.cos(lepton.phi)
    lepton_y = lepton.pt * np.sin(lepton.phi)
    lepton_z = lepton.pt * np.sinh(lepton.eta)

    # Convert jet kinematics from (pt, eta, phi) to cartesian components.
    jet_x = jet.pt * np.cos(jet.phi)
    jet_y = jet.pt * np.sin(jet.phi)
    jet_z = jet.pt * np.sinh(jet.eta)

    dot = lepton_x * jet_x + lepton_y * jet_y + lepton_z * jet_z
    jet_p2 = jet_x**2 + jet_y**2 + jet_z**2
    lepton_p2 = lepton_x**2 + lepton_y**2 + lepton_z**2

    # Compute pt_rel^2 = |p_lepton|^2 - (dot/|p_jet|)^2.
    safe_jet_p2 = ak.where(jet_p2 == 0, 1, jet_p2)
    pt_rel_sq = lepton_p2 - (dot**2) / safe_jet_p2
    pt_rel_sq = ak.where(pt_rel_sq < 0, 0, pt_rel_sq)

    return np.sqrt(pt_rel_sq)


def compute_mass(lepton, jet):
    """
    Compute the mass of the sum of lepton and the jet

    Parameters:
      lepton: a vector or Awkward array of lepton 4-vectors with fields
              "pt", "eta", "phi", "mass", having the same shape as `jet`.
      jet:    a vector or Awkward array of jet 4-vectors with fields
              "pt", "eta", "phi", "mass", having the same shape as `lepton`.

    Returns:
      An array (e.g. an Awkward array or NumPy array) of mass values.
    """

    # Energy + momentum components
    E_lepton = np.sqrt(lepton.pt**2 * np.cosh(lepton.eta) ** 2 + lepton.mass**2)
    px_lepton = lepton.pt * np.cos(lepton.phi)
    py_lepton = lepton.pt * np.sin(lepton.phi)
    pz_lepton = lepton.pt * np.sinh(lepton.eta)

    E_jet = np.sqrt(jet.pt**2 * np.cosh(jet.eta) ** 2 + jet.mass**2)
    px_jet = jet.pt * np.cos(jet.phi)
    py_jet = jet.pt * np.sin(jet.phi)
    pz_jet = jet.pt * np.sinh(jet.eta)

    # Invariant mass
    E_sum = E_lepton + E_jet
    px_sum = px_lepton + px_jet
    py_sum = py_lepton + py_jet
    pz_sum = pz_lepton + pz_jet

    mass2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
    mass2 = ak.where(mass2 < 0, 0, mass2)
    mass = np.sqrt(mass2)

    return mass


def delta_phi(p1, p2):
    """Vectorised delta phi between two arrays of angles."""
    dphi = np.abs(p1.phi - p2.phi)
    return ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)


def delta_eta(p1, p2):
    """Vectorised delta R between two sets of (eta,phi)."""
    return np.abs(p1.eta - p2.eta)


# ----------------------------
# Processor definition
# ----------------------------
class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=True,
        noHist=False,
        chunksize=75000,
        selectionModifier="",
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        # Substring matching (not equality) so modifiers can be COMBINED, e.g.
        # selectionModifier="lowpt_samesign". Exact equality silently dropped the
        # low-pT setting whenever a second modifier was appended. Backward
        # compatible: the only values used previously were "" and "lowpt".
        self.do_lowpt = "lowpt" in selectionModifier
        # Same-sign control region: inverts the e-mu charge requirement to select
        # a fake-lepton enriched sample for a data-driven background estimate.
        self.do_samesign = "samesign" in selectionModifier
        # settings for output
        #
        # The array output directory is resolved per worker. Autodetection is
        # unreliable on batch nodes: if neither the EOS nor the dust mount is
        # visible (or $USER is unset), the old code silently fell back to the
        # worker's cwd, scattering arrays into the submission directory while
        # the job still reported success.
        #
        # SFB_ARRAY_BASE pins the destination explicitly and is the recommended
        # way to run on HTCondor/dask. SFB_REQUIRE_SHARED_OUTPUT=1 additionally
        # turns the cwd fallback into a hard error, so a missing mount fails the
        # job instead of producing arrays nobody notices.
        username = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        explicit_base = os.environ.get("SFB_ARRAY_BASE", "").strip()
        require_shared = os.environ.get(
            "SFB_REQUIRE_SHARED_OUTPUT", ""
        ).lower() in ("1", "true", "yes", "on")

        cern_eos_base = f"/eos/user/{username[0]}/{username}" if username else ""
        desy_dust_base = f"/data/dust/user/{username}" if username else ""

        if explicit_base:
            # Explicit wins; create it so a typo surfaces here rather than as a
            # silently misplaced array much later.
            base_path = explicit_base
            os.makedirs(base_path, exist_ok=True)
            shared_output = True
        elif cern_eos_base and os.path.exists(cern_eos_base):
            base_path = cern_eos_base
            shared_output = True
        elif desy_dust_base and os.path.exists(desy_dust_base):
            base_path = desy_dust_base
            shared_output = True
        else:
            base_path = os.getcwd()
            shared_output = False

        if not shared_output:
            msg = (
                "sf_ttdilep_kin: no shared output area found "
                f"(USER={username!r}, tried {cern_eos_base!r} and "
                f"{desy_dust_base!r}). Arrays would be written under the "
                f"current directory ({base_path}) instead of the shared area. "
                "Set SFB_ARRAY_BASE to the desired base path."
            )
            if require_shared:
                raise RuntimeError(msg)
            print(f"WARNING: {msg}")

        self.out_dir_base = os.path.join(
            base_path,
            "btv/phys_btag/sfb-ttkinfit/arrays"
            + ("_lowpt" if self.do_lowpt else "")
            + ("_samesign" if self.do_samesign else ""),
        )  # noqa
        print(f"sf_ttdilep_kin: array output base = {self.out_dir_base}")

        ## Load corrections
        self.SF_map = load_SF(self._year, self._campaign)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        if len(events) == 0:
            return {}
        events = missing_branch(events)
        sumws = reweighting(events, self.isSyst)
        vetoed_events, shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(vetoed_events, collections), sumws, name)
            for collections, name in shifts
        )

    def process_shift(self, events, sumws, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")

        output = {"": None}
        if not self.noHist:
            output = histogrammer(
                events.Jet.fields,
                obj_list=["dilep"],
                hist_collections=["common", "fourvec", "ttdilep_kin"],
            )

        if shift_name is None:
            output["sumw"] = sumws["sumw"]
            if not isRealData and self.isSyst:
                if "LHEPdfWeight" in events.fields:
                    output["PDF_sumwUp"] = sumws["PDF_sumwUp"]
                    output["PDF_sumwDown"] = sumws["PDF_sumwDown"]
                    output["aS_sumwUp"] = sumws["aS_sumwUp"]
                    output["aS_sumwDown"] = sumws["aS_sumwDown"]
                    output["PDFaS_sumwUp"] = sumws["PDFaS_sumwUp"]
                    output["PDFaS_sumwDown"] = sumws["PDFaS_sumwDown"]
                if "LHEScaleWeight" in events.fields:
                    output["muR_sumwUp"] = sumws["muR_sumwUp"]
                    output["muR_sumwDown"] = sumws["muR_sumwDown"]
                    output["muF_sumwUp"] = sumws["muF_sumwUp"]
                    output["muF_sumwDown"] = sumws["muF_sumwDown"]
                if "PSWeight" in events.fields:
                    if len(events.PSWeight[0]) == 4:
                        output["ISR_sumwUp"] = sumws["ISR_sumwUp"]
                        output["ISR_sumwDown"] = sumws["ISR_sumwDown"]
                        output["FSR_sumwUp"] = sumws["FSR_sumwUp"]
                        output["FSR_sumwDown"] = sumws["FSR_sumwDown"]

        # ----------------------------
        # Basic Event Selection
        # ----------------------------
        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        ## HLT
        # 2016 (pre- and post-VFP) only has the Mu23_Ele12 and Mu8_Ele23 emu
        # cross triggers, in their non-DZ form (the Mu12_Ele23 path and the _DZ
        # variants were introduced in 2017). Using the non-DZ paths covers the
        # full 2016 dataset.
        if self._campaign in ["2016preVFP-UL", "2016postVFP-UL"]:
            triggers = [
                "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
            ]
        else:
            triggers = [
                "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            ]
        req_trig = HLT_helper(events, triggers)

        ## Primary vertex
        req_vtx = events.PV.npvs > 0

        # ----------------------------
        # Lepton Selections
        # ----------------------------
        # Muons: pt > 25, |eta| < 2.4, tightId, pfRelIso04_all < 0.15
        muons = events.Muon
        muon_sel = (muons.pt > 25) & mu_idiso(events, self._campaign)
        muons = muons[muon_sel]
        # For simplicity, we take the first muon in each event
        mu0 = ak.pad_none(muons, 1, clip=True)[:, 0]

        # Electrons: pt > 25, |(eta + deltaEtaSC)| < 2.4,  cutBased >= 3
        electrons = events.Electron
        ele_sel = (
            (electrons.pt > 25)
            & electrons.convVeto
            & ele_cuttightid(events, self._campaign)
        )
        electrons = electrons[ele_sel]
        # For simplicity, we take the first electron in each event
        ele0 = ak.pad_none(electrons, 1, clip=True)[:, 0]

        # For ttbar dilepton selection, require exactly 1 electron and 1 muon
        n_ele = ak.num(electrons)
        n_mu = ak.num(muons)
        req_dilepton = (n_ele == 1) & (n_mu == 1)

        dilepton = ele0 + mu0
        mll = ak.fill_none(dilepton.mass, -1)

        # Select events with either Z window removed or mll > 90 GeV to remove DY events
        # req_mll = np.abs(mll - 91.) > 15.
        req_mll = mll > 90.0

        # ----------------------------
        # Jet selection
        # ----------------------------
        jets = events.Jet
        jet_pt_threshold = 30.0 if not self.do_lowpt else 15.0
        # Apply jet cuts defined in jet_id (see utils/selection.py)
        jet_sel = ak.fill_none(
            jet_id(events, self._campaign, min_pt=jet_pt_threshold)
            & (abs(jets.eta) < 2.5)
            & (jets.pt > jet_pt_threshold)
            & (
                ak.all(
                    jets.metric_table(events.Muon) > 0.4,
                    axis=2,
                    mask_identity=True,
                )
            )
            & (
                ak.all(
                    jets.metric_table(events.Electron) > 0.4,
                    axis=2,
                    mask_identity=True,
                )
            ),
            False,
        )
        jets = jets[jet_sel]
        # Jet multiplicity
        jetmult = ak.num(jets)
        # For further computations, require at least two jets per event
        req_jets = jetmult > 1

        # ----------------------------
        # Event selection
        # ----------------------------
        # Electric-charge requirement: opposite-sign for the measurement region,
        # same-sign for the fake-enriched control region (do_samesign).
        _charge_sign = 1 if self.do_samesign else -1
        req_opposite_charge = (ele0.charge * mu0.charge) == _charge_sign
        req_opposite_charge = ak.fill_none(req_opposite_charge, False)

        # Missing transverse momentum > 40 GeV
        req_met = events.MET.pt > 40.0  # GeV

        event_level = (
            req_trig
            & req_lumi
            & req_vtx
            & req_dilepton
            & req_jets
            & req_opposite_charge
            & req_mll
            & req_met
        )
        event_level = ak.fill_none(event_level, False)

        if len(events[event_level]) == 0:
            if self.isArray:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    None,
                    ["nominal"],
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}

        # ----------------------------
        # Object selection
        # ----------------------------
        # Apply event-level selection
        pruned_ev = events[event_level]

        ev_jets = jets[event_level]
        ev_electrons = electrons[event_level]
        ev_muons = muons[event_level]

        # Extract the unique electron and muon per event
        ele = ak.firsts(ev_electrons)
        mu = ak.firsts(ev_muons)

        # ----------------------------
        # Dilepton pairing and channel assignment
        # ----------------------------
        # Assign channel (for eµ channel, following: ttbar_chan = (-11*charge_e) * (-13*charge_mu))
        ttbar_chan = (-11 * ele.charge) * (-13 * mu.charge)

        # Build the dilepton four‐vector
        dilepton = ev_electrons[:, 0] + ev_muons[:, 0]

        # ----------------------------
        # Compute Kinematic Variables (repeat for each jet in ev_jets)
        # ----------------------------
        # for each jet, compute delta_R to ele
        dR_ele_jet = ev_jets.delta_r(ele)
        dR_mu_jet = ev_jets.delta_r(mu)

        is_ele_close = dR_ele_jet < dR_mu_jet
        lepton_close = ak.where(is_ele_close, ele, mu)
        lepton_far = ak.where(is_ele_close, mu, ele)

        # Compute “far” system kinematics per jet:
        close_mlj = compute_mass(lepton_close, ev_jets)
        close_ptrel = compute_pt_rel(lepton_close, ev_jets)
        close_deta = delta_eta(ev_jets, lepton_close)
        close_dphi = delta_phi(ev_jets, lepton_close)
        close_lj2ll_deta = delta_eta(lepton_close, dilepton[:, None])
        close_lj2ll_dphi = delta_phi(lepton_close, dilepton[:, None])

        # Compute “far” system kinematics per jet:
        far_mlj = compute_mass(lepton_far, ev_jets)
        far_ptrel = compute_pt_rel(lepton_far, ev_jets)
        far_deta = delta_eta(ev_jets, lepton_far)
        far_dphi = delta_phi(ev_jets, lepton_far)
        far_lj2ll_deta = delta_eta(lepton_far, dilepton[:, None])
        far_lj2ll_dphi = delta_phi(lepton_far, dilepton[:, None])

        # Difference between jet and dilepton system:
        j2ll_deta = delta_eta(ev_jets, dilepton[:, None])
        j2ll_dphi = delta_phi(ev_jets, dilepton[:, None])

        # Compute a kinematic discriminator
        max_jets = int(ak.max(ak.num(ev_jets)))

        def pad_jet_var(var, max_jets=max_jets):
            var_reg = ak.pad_none(var, target=max_jets, axis=1, clip=True)
            return ak.to_numpy(ak.fill_none(var_reg, np.nan))

        bdt_event_hash = np.repeat(ak.to_numpy(pruned_ev.event), max_jets).reshape(
            -1, max_jets
        )
        padded_jetrank = pad_jet_var(ak.argsort(ak.argsort(-ev_jets.pt)))

        # Compute per-jet tagger discriminants (fill with NaN if not available)
        def get_tagger(jets, tag):
            return jets[tag] if tag in jets.fields else ak.full_like(jets.pt, np.nan)

        # list of available taggers
        taggers = {}
        disc_list = get_discriminators()
        for tag in disc_list:
            if tag in ev_jets.fields:
                taggers[tag] = get_tagger(ev_jets, tag)

        if not isRealData:
            # old implementation from TTBarCalib code
            # flavour = ak.where(
            #     ev_jets.hadronFlavour != 0,
            #     ev_jets.hadronFlavour,
            #     ak.where(
            #         (abs(ev_jets.partonFlavour) == 4) | (abs(ev_jets.partonFlavour) == 5),
            #         ak.zeros_like(ev_jets.pt, dtype=int),
            #         ev_jets.partonFlavour,
            #     ),
            # )
            # new implementation more in line with rest of commissioning framework
            flavour = ak.where(
                (ev_jets.partonFlavour == 0) & (ev_jets.hadronFlavour == 0),
                ak.ones_like(ev_jets.pt, dtype=int),
                ev_jets.hadronFlavour,
            )
            flavour = ak.values_astype(flavour, int)
        else:
            flavour = ak.zeros_like(ev_jets.pt, dtype=int)

        # Configure SFs
        weights = weight_manager(
            pruned_ev,
            self.SF_map,
            self.isSyst,
            ttbar_reweights=getattr(self, "ttbar_reweights", "none"),
            campaign=self._campaign,
        )
        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]

        # Output arrays
        if self.isArray:
            pruned_ev_array = pruned_ev
            # --- Assign general event info ---
            pruned_ev_array["EventInfo"] = np.repeat(
                np.column_stack(
                    [
                        ak.to_numpy(pruned_ev.run),
                        ak.to_numpy(pruned_ev.event),
                        ak.to_numpy(pruned_ev.luminosityBlock),
                    ]
                ),
                max_jets,
            ).reshape(-1, max_jets, 3)
            pruned_ev_array["ttbar_chan"] = np.repeat(
                ak.to_numpy(ttbar_chan), max_jets
            ).reshape(-1, max_jets)
            pruned_ev_array["npvn"] = np.repeat(
                ak.to_numpy(pruned_ev.PV.npvs), max_jets
            ).reshape(-1, max_jets)
            pruned_ev_array["jetmult"] = np.repeat(
                ak.to_numpy(ak.num(ev_jets)), max_jets
            ).reshape(-1, max_jets)
            pruned_ev_array["bdt_event_hash"] = bdt_event_hash
            pruned_ev_array["jetrank"] = padded_jetrank
            pruned_ev_array["jetpt"] = ev_jets.pt
            pruned_ev_array["jeteta"] = ev_jets.eta
            pruned_ev_array["flavour"] = pad_jet_var(flavour)
            # --- Assign the kin tree variables ---
            pruned_ev_array["close_mlj"] = pad_jet_var(close_mlj)
            pruned_ev_array["close_deta"] = pad_jet_var(close_deta)
            pruned_ev_array["close_dphi"] = pad_jet_var(close_dphi)
            pruned_ev_array["close_ptrel"] = pad_jet_var(close_ptrel)
            pruned_ev_array["close_lj2ll_deta"] = pad_jet_var(close_lj2ll_deta)
            pruned_ev_array["close_lj2ll_dphi"] = pad_jet_var(close_lj2ll_dphi)
            pruned_ev_array["far_mlj"] = pad_jet_var(far_mlj)
            pruned_ev_array["far_deta"] = pad_jet_var(far_deta)
            pruned_ev_array["far_dphi"] = pad_jet_var(far_dphi)
            pruned_ev_array["far_ptrel"] = pad_jet_var(far_ptrel)
            pruned_ev_array["far_lj2ll_deta"] = pad_jet_var(far_lj2ll_deta)
            pruned_ev_array["far_lj2ll_dphi"] = pad_jet_var(far_lj2ll_dphi)
            pruned_ev_array["j2ll_deta"] = pad_jet_var(j2ll_deta)
            pruned_ev_array["j2ll_dphi"] = pad_jet_var(j2ll_dphi)
            # --- Assign the tagger variables ---
            for tag, tag_var in taggers.items():
                pruned_ev_array[tag] = pad_jet_var(tag_var)

            array_writer(
                self,
                pruned_ev_array,
                events,
                weights,
                systematics,
                dataset,
                isRealData,
                self.out_dir_base + "/" if self.out_dir_base else "",
                kinOnly=[],
                othersData=[],
            )

        # Configure histograms
        if not self.noHist:
            pruned_ev_hist = pruned_ev
            pruned_ev_hist["SelJet"] = ev_jets[:, 0]
            pruned_ev_hist["SelMuon"] = ev_muons[:, 0]
            pruned_ev_hist["SelElectron"] = ev_electrons[:, 0]
            pruned_ev_hist["njet"] = ak.num(ev_jets)
            pruned_ev_hist["flavour"] = flavour
            pruned_ev_hist["close_mlj"] = close_mlj
            pruned_ev_hist["close_deta"] = close_deta
            pruned_ev_hist["close_dphi"] = close_dphi
            pruned_ev_hist["close_ptrel"] = close_ptrel
            pruned_ev_hist["close_lj2ll_deta"] = close_lj2ll_deta
            pruned_ev_hist["close_lj2ll_dphi"] = close_lj2ll_dphi
            pruned_ev_hist["far_mlj"] = far_mlj
            pruned_ev_hist["far_deta"] = far_deta
            pruned_ev_hist["far_dphi"] = far_dphi
            pruned_ev_hist["far_ptrel"] = far_ptrel
            pruned_ev_hist["far_lj2ll_deta"] = far_lj2ll_deta
            pruned_ev_hist["far_lj2ll_dphi"] = far_lj2ll_dphi
            pruned_ev_hist["j2ll_deta"] = j2ll_deta
            pruned_ev_hist["j2ll_dphi"] = j2ll_dphi

            output = histo_writter(
                pruned_ev_hist, output, weights, systematics, self.isSyst, self.SF_map
            )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
