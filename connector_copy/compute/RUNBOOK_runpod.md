# RUNBOOK — RunPod Setup

**Backend:** RunPod (fallback compute target, one-step pathway)
**Verified by:** Vishak Shashikumar, 2026-10-05
**Audience:** Anyone on the team (including bio students) who needs to run the agentic pipeline
on a RunPod GPU instead of HPC/Explorer.

This runbook assumes no prior RunPod experience. Follow it top to bottom in order —
several steps must happen in a specific sequence (noted where it matters) or you'll hit
avoidable errors covered in the Troubleshooting section at the end.

---

## 1. Create a RunPod account

1. Go to **https://www.runpod.io**
2. Click **Sign Up** → sign up with email or Google
3. Verify your email if prompted
4. When you land on "Onboard your AI agent to RunPod," click **Skip for now** (this is an
   optional AI-agent auto-setup feature — not needed for this runbook)

---

## 2. Add billing

1. Top right of the dashboard, click **Load credits**
2. Add a payment method and load at least **$10–20** (RunPod is prepaid; $10 is the minimum)
3. Confirm your balance shows the loaded amount

**Cost reference:** A100 SXM/PCIe (80GB) runs **~$1.59/hr**. A full environment setup +
smoke test typically uses well under $5 of compute.

---

## 3. Generate an SSH key pair on your own computer

Do this in your local terminal (Mac/Linux Terminal, or Git Bash on Windows) — **not** inside
RunPod yet.

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

- Press **Enter** at the file-location prompt (accept the default)
- Press **Enter** twice more for an empty passphrase (no passphrase — a passphrase would
  block any automated/scripted SSH use later)

Then print your public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output line (starts with `ssh-ed25519 AAAA...`).

---

## 4. Add your SSH key to RunPod — do this BEFORE deploying any pod

> **This step must happen before you deploy a pod.** Adding a key to your account after a
> pod is already running will **not** retroactively add it to that pod's `authorized_keys`.
> If you forget this step, see Troubleshooting #1.

1. In RunPod's left sidebar, go to **Credentials**
2. Click the **SSH Public Keys** tab
3. Click **+ Add SSH key**
4. **Title:** anything identifiable, e.g. `yourname-laptop`
5. **Key:** paste the full public key line from Step 3
6. Confirm it appears in the list with a fingerprint and today's date

---

## 5. Deploy a pod

1. Left sidebar → **Pods** → **+ Deploy**
2. Select the **Pods** card (not Serverless)
3. Choose the **PyTorch 2.8.0** template ("Most Popular") — prebuilt PyTorch + CUDA, good
   general-purpose base for this project
4. Click **See all GPUs**, and pick an **80GB-class GPU** (A100 SXM or A100 PCIe,
   ~$1.59/hr). **Check the "Available" tab**, not "All" — GPU availability shifts
   constantly; only deploy what's actually shown as available right now, not what's
   queued/unavailable
5. Set **GPU count: 1**
6. Under **Storage → Persistent storage**, click **Add volume** and select **Volume disk**
   (not Network volume — see Troubleshooting #2 for why), set size to **50GB**
7. Confirm the mount path shows `/workspace`
8. Give the pod a clear name (top of the Deploy screen, or rename after deploying)
9. Click **Deploy Pod**

Wait 1–2 minutes for "Initializing your pod..." to finish and show as running (green dot).

---

## 6. Connect via SSH

1. Click into the running pod → **Connect** tab
2. Scroll to **"SSH over exposed TCP"** (use this one, not the first "SSH" option — this
   one supports SCP/SFTP for file transfer, the other doesn't)
3. Copy the command shown, e.g.:
   ```bash
   ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
   ```
4. Run it in your local terminal
5. Type `yes` if asked to confirm the host fingerprint
6. You should land directly in a root shell — **no password prompt**. If it asks for a
   password, see Troubleshooting #1.

---

## 7. Verify the environment

Once connected, run each of these and confirm sane output:

```bash
nvidia-smi
```
Should show your GPU (e.g. `NVIDIA A100-SXM4-80GB`) with 0MiB used.

```bash
pwd && ls /workspace
```
Confirms your persistent volume is mounted (will be empty on first boot — that's correct).

```bash
df -h
```
Confirms `/workspace` shows your volume size (not just the 30GB container disk).

```bash
python3 --version
nvcc --version
```
Confirms Python and CUDA toolkit versions.

---

## 8. Clone the group repo

The repo is private, so you need a **GitHub Personal Access Token** — a regular password
will **not** work for Git operations (GitHub blocks it).

**Generate a token (in your browser, not the pod terminal):**
1. Go to **https://github.com/settings/tokens**
2. **Generate new token** → **Generate new token (classic)**
3. Note: anything identifiable, e.g. `runpod-setup`
4. Check the **`repo`** scope
5. Generate, and **copy the token immediately** (shown only once, starts with `ghp_...`)

**Clone, with the token embedded directly in the URL** (avoids broken username/password
prompts entirely):

```bash
cd /workspace
git clone https://<YOUR_TOKEN>@github.com/sinitskiy/reproducing_Complexa_paper_Fall2026.git
cd reproducing_Complexa_paper_Fall2026
```

---

## 9. Set up the connector

```bash
export RUNPOD_API_KEY="your_real_runpod_api_key"
```

Get your real API key from RunPod → **Settings → API Keys** (different from the SSH key —
this is a separate credential used by the Python `runpod` package to talk to RunPod's API).

```bash
pip install runpod
```

If you hit a `cryptography` package conflict error during install, run:
```bash
pip install --ignore-installed cryptography
pip install runpod --break-system-packages
```

```bash
cd compute
cp connector_config.example.json connector_config.json
nano connector_config.json
```

In the `"runpod"` section, change:
```json
"gpu_type": "A100",
```
to the **exact GPU API id string**, not the display name:
```json
"gpu_type": "NVIDIA A100-SXM4-80GB",
```
(Match whatever GPU you actually deployed — check `nvidia-smi`'s output for the exact name.)

Save (`Ctrl+O`, `Enter`) and exit (`Ctrl+X`).

**Generate a second SSH key, this time directly on the pod itself** (separate from the key
you used to connect from your laptop — the connector needs its own key pair local to the
pod):
```bash
ssh-keygen -t ed25519
```
Accept all defaults (empty passphrase).

---

## 10. Verify the connector works

```bash
python3 connector.py --runner runpod check
```

Expected output:
```
config : /workspace/.../compute/connector_config.json
runner : runpod
target : runpod  gpu=NVIDIA A100-SXM4-80GB  image=None  cloud=SECURE
OK
```

If you see `OK`, the connector is fully verified and ready to use.

---

## 11. Shut down when done

**Always stop your pod when you're finished for the day** — billing is per-second while
running.

1. Go to **Pods**, click the pod
2. Click **Stop**
3. Confirm — your Volume Disk data is preserved across stop/restart (only lost if you
   fully **delete** the pod, not stop it)

**Caution:** a stopped pod resumes on the *same physical host* it started on. If that
host's GPU capacity gets reallocated while stopped, **resume can fail** ("not enough free
GPUs on the host machine"). If this happens, your data on the Volume Disk is still safe,
but you'll need to deploy a fresh pod and re-clone/re-setup (repeat from Step 5, re-cloning
the repo again).

---

## Troubleshooting

**#1 — SSH asks for a password instead of connecting automatically**
Your key was added to RunPod *after* the pod was already deployed. RunPod only injects
account-level SSH keys into a pod at boot/deploy time, not retroactively into a running
pod. Fix: add your key (Step 4) **before** deploying, or deploy a fresh pod after adding
the key.

**#2 — Network Volume shows "No GPUs in region" and won't attach**
Network Volumes are pinned to one specific data center at creation. If the GPU you want
isn't currently available in that exact data center, the volume can't attach — even if
you just confirmed that GPU was available moments earlier (availability shifts constantly).
Use **Volume Disk** instead (Step 5) — it's provisioned as part of the pod itself, so it
always matches whatever GPU/region combination is actually live. Trade-off: Volume Disk is
tied to one pod and can't be shared across multiple pods the way a Network Volume can.

**#3 — `git clone` fails with "Password authentication is not supported for Git operations"**
You're trying to use your GitHub password, or the username field got mangled with extra
text. Use a Personal Access Token embedded directly in the clone URL (Step 8) instead of
answering separate username/password prompts.

**#4 — `pip install runpod` fails with "Cannot uninstall cryptography... no RECORD file"**
A system (Debian-packaged) version of `cryptography` conflicts with pip. Run:
```bash
pip install --ignore-installed cryptography
pip install runpod --break-system-packages
```

**#5 — `connector.py check` runs but `submit`/`logs`/`watch` behave unexpectedly**
The connector's config has a `"default"` runner (usually `"hpc"`). Some subcommands
validate the default runner's config even when you're targeting a different one. Always
pass `--runner runpod` explicitly on every command, not just the first one.

**#6 — "Pod resume failed: not enough free GPUs on the host machine"**
See the Shutdown caution above. Deploy a fresh pod; your Volume Disk data is safe, but
the pod environment (installed packages, etc.) needs to be rebuilt from Step 5 onward.
