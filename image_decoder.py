# app.py
# Streamlit demo: NSD fMRI → CLIP (text) → Stable Diffusion reconstruction
# Author: Nathaniel's workshop starter, 2026-02
# References:
# - NSD AWS Open Data bucket (selective download) https://registry.opendata.aws/nsd/  [1]
# - nsd_access helper for NSD paths and loaders https://github.com/tknapen/nsd_access [2]
# - CLIP (Hugging Face Transformers) https://huggingface.co/docs/transformers/model_doc/clip [3]
# - Diffusers (Stable Diffusion pipelines) https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/overview [4]
# - Takagi & Nishimoto (CVPR 2023) https://openaccess.thecvf.com/content/CVPR2023/papers/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.pdf [5]
# - Brain-Diffuser (Ozcelik & VanRullen, 2023) https://github.com/ozcelikfu/brain-diffuser [6]
# - Stable unCLIP (optional advanced variant) https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_unclip [7]

import os
import io
import json
import time
import numpy as np
import pandas as pd
import nibabel as nib
import h5py
from PIL import Image
from typing import List, Tuple

import streamlit as st
from tqdm import tqdm

import torch
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# CLIP (text & image encoders)
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel

# Stable Diffusion (text-to-image). You can switch to SDXL or SD-2.1-unCLIP below.
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# NSD helper
from nsd_access.nsda import NSDAccess


# -----------------------
# Streamlit Page config
# -----------------------
st.set_page_config(page_title="NSD: CLIP→Stable Diffusion Reconstruction", layout="wide")
st.title("NSD fMRI → CLIP → Stable Diffusion (Streamlit Demo)")

st.markdown("""
This app decodes a small subset of NSD single-trial **fMRI betas** into **CLIP text embeddings**, retrieves the nearest **COCO captions**, and uses **Stable Diffusion** to reconstruct a plausible scene.
- Data: Natural Scenes Dataset (**NSD**). Download a few sessions for one subject + stimuli and metadata (see side panel).  
- Models: **CLIP** (Hugging Face Transformers) and **Stable Diffusion** (Hugging Face Diffusers).  
- Method: Simple **ridge regression** decoder for speed & transparency.  
""")

# -----------------------
# Sidebar: configuration
# -----------------------
with st.sidebar:
    st.header("Configuration")
    NSD_ROOT = st.text_input("NSD root folder", value="NSD")
    subject = st.text_input("Subject (e.g., subj01)", value="subj01")
    sessions = st.text_input("Sessions (comma-separated)", value="1,2")
    sessions = [int(s.strip()) for s in sessions.split(",") if s.strip()]

    max_trials = st.number_input("Max trials to use", min_value=50, max_value=2000, value=400, step=50)
    test_fraction = st.slider("Test fraction", 0.05, 0.5, 0.2, 0.05)

    seed = st.number_input("Random seed", min_value=0, value=0, step=1)
    steps = st.slider("Stable Diffusion steps", 10, 75, 35, 5)
    guidance = st.slider("CFG guidance scale", 1.0, 15.0, 7.5, 0.5)

    use_gpu = st.checkbox("Use GPU if available", value=True)

    st.markdown("**Data prerequisites**")
    st.code("""
aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/
aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata/ppdata/subj01/func1pt8mm/brainmask.nii.gz NSD/nsddata/ppdata/subj01/func1pt8mm/
aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv NSD/nsddata/experiments/nsd/
aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 NSD/nsddata_stimuli/stimuli/nsd/
""", language="bash")

    st.markdown("**COCO captions (needed for prompt retrieval)**")
    st.code("""
# one-liner via nsd_access helper (downloads captions_trainval2017.zip and extracts)
# in Python: nsda.download_coco_annotation_file()
""", language="bash")

    advanced_box = st.expander("Advanced: Try SD-2.1-unCLIP (image embedding route)")
    with advanced_box:
        use_unclip = st.checkbox("Use Stable unCLIP (experimental)", value=False)
        unclip_model_id = st.text_input("unCLIP model (img-variation)", value="stabilityai/stable-diffusion-2-1-unclip-small")
        unclip_noise = st.slider("unCLIP noise_level", 0, 400, 50, 10)
        st.caption("unCLIP conditions on CLIP image embeddings for image variations. Great for an advanced demo, but heavier. [Docs]")

# -----------------------
# Caching: devices & models
# -----------------------
@st.cache_resource(show_spinner=False)
def get_device(use_gpu_flag: bool):
    if use_gpu_flag and torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = get_device(use_gpu)

@st.cache_resource(show_spinner=True)
def load_clip(model_name="openai/clip-vit-base-patch32", device="cpu"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    proc = CLIPProcessor.from_pretrained(model_name)
    tok = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)
    return model, proc, tok, text_model

@st.cache_resource(show_spinner=True)
def load_sd_pipeline(model_id="runwayml/stable-diffusion-v1-5", device="cpu"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    if device == "cuda":
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

# OPTIONAL: unCLIP pipeline loader (commented since we default to text prompt route)
# from diffusers import StableUnCLIPImg2ImgPipeline
# @st.cache_resource(show_spinner=True)
# def load_unclip_pipeline(model_id, device="cpu"):
#     pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
#         model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32
#     )
#     if device == "cuda":
#         pipe = pipe.to("cuda")
#         pipe.enable_attention_slicing()
#     return pipe

clip_model, clip_proc, clip_tok, clip_text_model = load_clip(device=device)
sd_pipe = load_sd_pipeline(device=device)
# unclip_pipe = load_unclip_pipeline(unclip_model_id, device=device) if use_unclip else None


# -----------------------
# NSD loading utilities
# -----------------------
def load_nsd_subset(nsd_root: str, subj: str, sess_list: List[int], max_n: int):
    """
    Returns:
        betas: [N_trials, n_vox]
        trial_df: dataframe with at least columns [nsd_id, cocoId, session]
        imgs: list[PIL.Image] for those nsd_id
        mask: boolean 3D brain mask array
    """
    nsda = NSDAccess(nsd_root)

    # Read brainmask
    brainmask_path = os.path.join(nsd_root, "nsddata", "ppdata", subj, "func1pt8mm", "brainmask.nii.gz")
    mask = nib.load(brainmask_path).get_fdata().astype(bool)

    # Read betas (GLMdenoise RR single-trial) per session & stack
    beta_4ds, rows = [], []
    for sess in sess_list:
        nii = nsda.read_betas(subject=subj,
                              data_format="func1pt8mm",
                              betas_version="betas_fithrf_GLMdenoise_RR",
                              sess=sess)
        beta_4ds.append(nii.get_fdata())  # (X,Y,Z,T)
        stim_info_path = os.path.join(nsd_root, "nsddata", "experiments", "nsd", "nsd_stim_info_merged.csv")
        df = pd.read_csv(stim_info_path)
        sess_df = df.query("subject == @subj and session == @sess").copy()
        # normalize column name to 'nsd_id'
        if "nsd_id" not in sess_df.columns and "stimulus_id" in sess_df.columns:
            sess_df = sess_df.rename(columns={"stimulus_id": "nsd_id"})
        rows.append(sess_df)

    trial_df = pd.concat(rows, ignore_index=True)
    # To keep the workshop fast, truncate:
    if len(trial_df) > max_n:
        trial_df = trial_df.iloc[:max_n].copy()

    # Shape betas into [N, V]
    betas_list = []
    sofar = 0
    for data_4d, sess in zip(beta_4ds, sess_list):
        T = data_4d.shape[-1]
        # pick trials belonging to this session and within max_n slice
        mask_rows = (trial_df["session"] == sess).values
        n_sess_trials = mask_rows.sum()
        if n_sess_trials == 0:
            continue
        # linearize & select trials
        data_2d = data_4d[mask].T  # [T, V]
        pick = min(n_sess_trials, data_2d.shape[0])
        betas_list.append(data_2d[:pick, :])

    betas = np.vstack(betas_list)

    # Load images from HDF5 by nsd_id
    imgs = []
    h5_path = os.path.join(nsd_root, "nsddata_stimuli", "stimuli", "nsd", "nsd_stimuli.hdf5")
    with h5py.File(h5_path, "r") as f:
        img_ds = f["imgBrick"]
        for i in tqdm(trial_df["nsd_id"].astype(int).tolist(), desc="Reading images"):
            arr = img_ds[i]  # (H, W, 3) uint8
            imgs.append(Image.fromarray(arr))

    return betas, trial_df, imgs, mask


@st.cache_resource(show_spinner=True)
def load_captions_from_coco(nsd_root: str) -> pd.DataFrame:
    """
    Load COCO captions annotations shipped with NSD stimuli.
    Returns a DataFrame with columns ['cocoId', 'caption'].
    """
    nsda = NSDAccess(nsd_root)
    # Ensure captions json exists; if not, the helper can download/extract
    # nsda.download_coco_annotation_file()
    ann_dir = os.path.join(nsd_root, "nsddata_stimuli", "stimuli", "nsd", "annotations")
    train_caps = os.path.join(ann_dir, "captions_train2017.json")
    val_caps = os.path.join(ann_dir, "captions_val2017.json")

    def read_caps(path):
        if not os.path.exists(path):
            return pd.DataFrame(columns=["image_id", "caption"])
        with open(path, "r") as f:
            jj = json.load(f)
        return pd.DataFrame(jj["annotations"])[["image_id", "caption"]]

    df_train = read_caps(train_caps)
    df_val = read_caps(val_caps)
    caps = pd.concat([df_train, df_val], ignore_index=True)
    caps = caps.rename(columns={"image_id": "cocoId"})
    return caps


# -----------------------
# Feature functions
# -----------------------
def clip_text_embed(texts: List[str]) -> np.ndarray:
    # Use CLIP text encoder via tokenizer+CLIPTextModel
    # We'll take last_hidden_state pooled by CLS token or use model's text features via CLIPModel
    inputs = clip_tok(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        txt_feats = clip_model.get_text_features(**inputs)
        txt_feats = torch.nn.functional.normalize(txt_feats, dim=-1)
    return txt_feats.cpu().numpy()

def clip_image_embed(pils: List[Image.Image]) -> np.ndarray:
    with torch.no_grad():
        inputs = clip_proc(images=pils, return_tensors="pt", padding=True).to(device)
        img_feats = clip_model.get_image_features(**inputs)
        img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
    return img_feats.cpu().numpy()


# -----------------------
# Main app workflow
# -----------------------
colL, colR = st.columns([1.2, 1.0], gap="large")

with colL:
    st.subheader("1) Load a small NSD subset")
    if st.button("Load NSD subset"):
        with st.spinner("Reading betas, images, and metadata..."):
            betas, trial_df, imgs, mask = load_nsd_subset(NSD_ROOT, subject, sessions, max_trials)
        st.success(f"Loaded betas: {betas.shape}, trials: {len(trial_df)}, images: {len(imgs)}")
        st.session_state["betas"] = betas
        st.session_state["trial_df"] = trial_df
        st.session_state["imgs"] = imgs

    if "betas" in st.session_state:
        st.write("Example stimuli:")
        excols = st.columns(6)
        for i in range(min(6, len(st.session_state["imgs"]))):
            excols[i].image(st.session_state["imgs"][i], caption=f"trial {i}", use_column_width=True)

with colR:
    st.subheader("2) Build a caption bank (COCO)")
    if st.button("Load COCO captions"):
        with st.spinner("Loading COCO captions (train/val) from NSD annotations..."):
            caps = load_captions_from_coco(NSD_ROOT)
        st.success(f"Loaded {len(caps)} captions")
        st.session_state["caps"] = caps


st.markdown("---")

st.subheader("3) Train brain→CLIP(text) decoder (ridge)")
st.caption("We decode to CLIP **text** embedding, then retrieve nearest captions to prompt Stable Diffusion. This follows LDM-based reconstruction logic used in prior work, adapted for a fast workshop demo. [1](https://openaccess.thecvf.com/content/CVPR2023/papers/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.pdf)[2](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v3)")

if st.button("Compute embeddings & train"):
    assert "betas" in st.session_state and "trial_df" in st.session_state and "imgs" in st.session_state, "Load NSD first."
    assert "caps" in st.session_state, "Load COCO captions first."

    rng = np.random.default_rng(seed)
    betas = st.session_state["betas"]
    trial_df = st.session_state["trial_df"]
    imgs = st.session_state["imgs"]
    caps_df = st.session_state["caps"]

    # 3a) Build a caption for each trial using cocoId mapping (choose first caption occurrence)
    cap_map = caps_df.groupby("cocoId")["caption"].first().to_dict()
    trial_caps = []
    for coco in trial_df["cocoId"].astype(int).tolist():
        trial_caps.append(cap_map.get(coco, "a natural scene"))

    # 3b) Create a caption bank (unique captions) for retrieval
    # You can expand this to all captions (may be large). Start with those tied to your trials:
    caption_bank = list(set(trial_caps))
    st.info(f"Caption bank size: {len(caption_bank)} (from trial COCO captions)")

    # 3c) Compute text embeddings for train targets & bank
    with st.spinner("Encoding captions with CLIP text encoder..."):
        Y_all = clip_text_embed(trial_caps)        # targets per trial
        bank_emb = clip_text_embed(caption_bank)   # retrieval bank
    X_all = betas

    # 3d) Split train/test chronologically
    n = len(X_all)
    split = int((1.0 - test_fraction) * n)
    X_tr, X_te = X_all[:split], X_all[split:]
    Y_tr, Y_te = Y_all[:split], Y_all[split:]

    # 3e) Fit ridge regression with standardization
    alphas = np.logspace(-2, 2, 7)
    decoder = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                            RidgeCV(alphas=alphas, fit_intercept=True))
    with st.spinner("Training ridge decoder..."):
        decoder.fit(X_tr, Y_tr)

    # Evaluate (retrieval accuracy among the caption bank)
    Y_pred = decoder.predict(X_te)
    Y_pred = Y_pred / np.linalg.norm(Y_pred, axis=1, keepdims=True)
    sims = cosine_similarity(Y_pred, bank_emb)
    hits1 = []
    # ground truth target captions (nearest in bank)
    gt_idxs = []
    for j, y in enumerate(Y_te):
        y_norm = y / np.linalg.norm(y)
        gt_idx = np.argmax(cosine_similarity(y_norm.reshape(1,-1), bank_emb))
        gt_idxs.append(gt_idx)
        hits1.append(np.argmax(sims[j]) == gt_idx)
    acc1 = np.mean(hits1)
    st.success(f"Caption retrieval (Top-1) vs bank: {acc1:.3f}")

    # Save state
    st.session_state.update({
        "decoder": decoder,
        "caption_bank": caption_bank,
        "bank_emb": bank_emb,
        "split_idx": split,
        "X_te": X_te,
        "trial_df": trial_df,
        "imgs": imgs
    })


st.markdown("---")

st.subheader("4) Reconstruct a held-out trial with Stable Diffusion")
st.caption("We predict a CLIP text embedding from fMRI for a held-out trial, retrieve the **top‑k captions** from the bank, concatenate them as a prompt, and render with **Stable Diffusion**. (You can swap in SDXL, change scheduler, or try unCLIP as an advanced exercise.) [6](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/overview)")

if "decoder" in st.session_state:
    idx_in_test = st.number_input("Pick a test index (0-based)", min_value=0, max_value=max(0, len(st.session_state["X_te"])-1), value=0, step=1)
    topk = st.slider("Top‑k captions to combine", 1, 5, 3, 1)

    colA, colB = st.columns([1,1], gap="large")

    with colA:
        if st.button("Decode & reconstruct"):
            decoder = st.session_state["decoder"]
            caption_bank = st.session_state["caption_bank"]
            bank_emb = st.session_state["bank_emb"]
            split = st.session_state["split_idx"]
            X_te = st.session_state["X_te"]
            trial_df = st.session_state["trial_df"]
            imgs = st.session_state["imgs"]

            # Predict CLIP text embedding
            y_pred = decoder.predict(X_te[idx_in_test:idx_in_test+1])
            y_pred = y_pred / np.linalg.norm(y_pred, axis=1, keepdims=True)

            # Nearest captions from bank form our prompt
            sims = cosine_similarity(y_pred, bank_emb)[0]
            nn_idx = np.argsort(-sims)[:topk]
            prompt_caps = [caption_bank[i] for i in nn_idx]
            prompt = " ".join(prompt_caps)

            # Generate with Stable Diffusion
            generator = torch.Generator(device if device=="cuda" else "cpu").manual_seed(seed)
            with torch.autocast("cuda", enabled=(device=="cuda")):
                image = sd_pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator).images[0]

            st.image(image, caption=f"Reconstruction (prompt from decoded CLIP text)", use_column_width=True)
            st.code(prompt)

    with colB:
        # Show the ground truth stimulus image (nearest test trial)
        trial_idx_global = st.session_state["split_idx"] + idx_in_test
        if "imgs" in st.session_state and trial_idx_global < len(st.session_state["imgs"]):
            st.image(st.session_state["imgs"][trial_idx_global], caption="Ground truth stimulus (for reference)", use_column_width=True)

    st.info("Tip: reduce 'top‑k' if prompts become too long. You can also pass a single best caption only.")

    # --- Experimental unCLIP (image-embedding route) ---
    if use_unclip:
        st.warning("Experimental: unCLIP path expects a CLIP image embedding, but the public pipeline typically extracts embeddings from an input image. "
                   "For a true brain→embedding→image path, you would inject y_pred into a custom unCLIP pipeline or retrieve a proxy image matching the decoded CLIP *image* embedding, then run image-variation.")
        st.caption("Docs: Stable unCLIP in diffusers (conditioning on CLIP image embeddings). [10](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_unclip)")
        # For teaching, a practical workaround is:
        # 1) Decode to CLIP *image* embeddings (clip_model.get_image_features on training images; learn fMRI->image embedding)
        # 2) Retrieve the nearest training image by cosine similarity as a proxy
        # 3) Feed that proxy image to StableUnCLIPImg2ImgPipeline as an 'image variation' seed
        # (Left as an exercise; commented out for simplicity and runtime.)
else:
    st.info("Train the decoder first (step 3).")


st.markdown("---")
st.subheader("5) Notes & references")

st.markdown("""
- **NSD access & selective download:** The dataset is mirrored on **AWS Open Data** for unauthenticated browsing/download (keep your subset small for a workshop). [8](https://registry.opendata.aws/nsd/)  
- **nsd_access helper:** one‑line accessors for betas, stimuli HDF5, and COCO annotations. [9](https://github.com/tknapen/nsd_access)  
- **CLIP models:** text & image embedders used here (`openai/clip-vit-base-patch32`). [5](https://huggingface.co/docs/transformers/en/model_doc/clip)  
- **Stable Diffusion (Diffusers):** text‑to‑image pipelines with schedulers (we use DPM‑Solver). [6](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/overview)  
- **Prior art:** fMRI→Stable Diffusion reconstructions (Takagi & Nishimoto, CVPR 2023; Brain‑Diffuser 2023). [1](https://openaccess.thecvf.com/content/CVPR2023/papers/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.pdf)[2](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v3)[3](https://github.com/yu-takagi/StableDiffusionReconstruction)[4](https://github.com/ozcelikfu/brain-diffuser)  
- **Stable unCLIP (advanced):** condition SD on CLIP **image** embeddings for image variation or with a prior for T2I. [10](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_unclip)[11](https://huggingface.co/docs/diffusers/v0.18.0/api/pipelines/stable_unclip)
""")