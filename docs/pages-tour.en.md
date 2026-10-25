[简体中文](pages-tour.md) | English

# RunOS pages tour

Screenshots were taken in an empty-data environment (no datasets, no
downloaded models, no compute nodes), signed in and captured page by page.
Pages that need a dataset show a "no dataset selected" placeholder when empty;
that is the normal empty state, not a fault.
The sign-in page uses the same light theme as the rest of the app.

## Sign-in page

Local admin sign-in. Demo account `admin / runos-demo`; change the password
in `.env` before exposing a deployment.

![Sign-in page](screenshots/login.png)

## Home

Opens the datasets view by default (the same view as the "Datasets" page).

![Home](screenshots/home.png)

## Datasets

Load and manage image collections: upload a ZIP or browse a local folder.
The empty environment shows the upload and load entries.

![Datasets](screenshots/datasets.png)

## Dashboard

Statistics, class distribution, and analysis insights for the current dataset.
Shows a placeholder when no dataset is selected.

![Dashboard](screenshots/dashboard.png)

## Gallery

Browse all images in a visual grid; clicking an image opens it in the
annotator. Shows a placeholder when no dataset is selected.

![Gallery](screenshots/gallery.png)

## Sorting

Review images one by one and organize data quickly with keyboard shortcuts.
Shows a placeholder when no dataset is selected.

![Sorting](screenshots/sorting.png)

## Duplicate detection

Automatically find and remove duplicate or highly similar images. Shows a
placeholder when no dataset is selected.

![Duplicate detection](screenshots/duplicate-detection.png)

## Annotate

Draw bounding boxes and labels on images to teach the AI what to recognize.
Shows a placeholder when no dataset is selected.

![Annotate](screenshots/annotate.png)

## Classes

Define the target classes the AI should learn (for example, car, person).
Shows a placeholder when no dataset is selected.

![Classes](screenshots/classes.png)

## Compute nodes (highlight)

Register several servers as compute nodes; training, inference, and cloud LLM
workloads run on them. This is how RunOS pools compute across servers.

- Add a node: connect over SSH or API, with host, port, username, password or
  key authentication, workdir, and Python environment (system / conda /
  venv).
- Automatic detection: after adding, it checks connectivity, OS, workdir, and
  GPU list, with errors in the report; first connection asks you to confirm
  the host fingerprint.
- Jump-host forwarding: servers on a private network can be reached through a
  public jump host (local → jump host → private server); the jump host also
  supports password or key authentication.
- Credential safety: compute-node credentials can only be managed from the
  local RunOS instance.

The node list is currently empty; the screenshot shows the add-node entry.

![Compute nodes](screenshots/compute-nodes.png)

## VLM configuration / LLM auto-annotation (highlight)

Connect a vision LLM, let it label data automatically, review the results,
and apply them to the dataset — a low-cost replacement for pure manual
annotation.

- Providers: Qwen-VL (Alibaba Cloud DashScope), OpenAI GPT-4o, local vLLM,
  and local Ollama, all through OpenAI-compatible APIs; plus a dedicated
  VLX-Seek configuration.
- Flow: configure the endpoint and classes → launch an auto-annotation job
  → pause, resume, or stop the job → review each annotation → apply to the
  dataset.
- Everything is viewable: annotation results, job logs, and raw outputs.

The screenshot shows an existing dedicated VLX-Seek configuration (model
VLX-Seek-1.5-10B); OpenAI-compatible models can be added with "New".

![VLM configuration](screenshots/vlm.png)

## Multimodal detection (highlight)

Describe the target in natural language and let the model return detection
conclusions and alerts directly — the entry to RunOS real-time multimodal
recognition.

- Describe to detect: for example, "find boats on the water"; the cloud
  VLX-Seek detector returns conclusions and raises alerts by rule.
- Evidence retention: tasks carry result JSON, result video, snapshots, and
  alert evidence (alert evidence can be recovered and viewed); video tasks,
  per-frame, and single-image inference are also available.
- Visible status: local YOLOE model status and VLX-Seek node connectivity
  are shown in the UI, with a clear notice when something is not configured.

No VLX-Seek compute node is configured yet; the screenshot shows the task
and configuration entries.

![Multimodal detection](screenshots/lmm.png)

## Train / validation / test split

Split a dataset into training, validation, and test sets so models are
evaluated correctly. Shows a placeholder when no dataset is selected.

![Train / validation / test split](screenshots/split.png)

## Augmentation

Automatically apply flips, rotations, and color changes to expand training
data. Shows a placeholder when no dataset is selected.

![Augmentation](screenshots/augmentation.png)

## Video frame extraction

Extract frames from video files to use as training material.

![Video frame extraction](screenshots/video-extraction.png)

## Format conversion

Convert datasets between YOLO, COCO, Pascal VOC, and other formats.

![Format conversion](screenshots/convert.png)

## Merge

Merge two or more separate datasets into one unified dataset.

![Merge](screenshots/merge.png)

## YAML wizard

Generate the configuration file needed to start model training. Shows a
placeholder when no dataset is selected.

![YAML wizard](screenshots/yaml-wizard.png)

## Model training

Train models on annotated datasets and track loss and accuracy live. The
screenshot shows the training configuration entry.

![Model training](screenshots/training.png)

## Inference

Test a trained model on images, video files, or a live camera, with object
tracking and ReID support.

![Inference](screenshots/inference.png)

## Evaluation

Compare several models side by side on mAP, precision, recall, and other
metrics.

![Evaluation](screenshots/evaluate.png)

## Models

View and manage trained models; run inference to test how well they work.
The model directory is currently empty (yolov8n not downloaded).

![Models](screenshots/models.png)

## Batch jobs

Monitor and manage long-running background jobs such as exports and
augmentations. No jobs are running right now.

![Batch jobs](screenshots/batch-jobs.png)

## Health check

Scan a dataset for quality problems such as missing labels or damaged
images. Shows a placeholder when no dataset is selected.

![Health check](screenshots/health.png)

## Compare

Compare two or more datasets side by side with statistics.

![Compare](screenshots/compare.png)

## Snapshots

Save dataset snapshots at any time and restore them when needed. Shows a
placeholder when no dataset is selected.

![Snapshots](screenshots/snapshots.png)

## Settings

Configure app preferences and the backend connection (API URL, storage
paths, GPU, preferences).
Note: the backend status badge shows "not connected" because the saved API
URL is empty (the `/api/proxy` in the form is a draft default); app data
actually goes through the `/api/proxy/api/*` proxy, which is verified
connected.

![Settings](screenshots/settings.png)
