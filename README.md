[English](README.en.md) | 简体中文

# RunOS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/melba818/RunOS-Drone-Platform)](https://github.com/melba818/RunOS-Drone-Platform/stargazers)

**无人机视觉数据，从照片到模型，一个服务全包。**

![首页](docs/screenshots/home.png)

无人机飞一趟，带回几千张照片、几小时视频——然后呢？数据散在硬盘里，标注靠人一张张点，显卡在别的机器上空转，工具换了一个又一个，模型还没跑起来，时间先花光了。

RunOS 就是来收拾这条链路的：数据集、标注、训练、推理，全在一个可独立部署的服务里跑通。它从 visos 长出来，留下了 visos 扎实的数据与标注底座，又长出了实时多模态识别和大模型自动标注。


## 快速开始

```bash
cp .env.example .env   # 先改掉演示密码

docker compose up -d --build
```

打开 http://127.0.0.1:19000 ，用 `.env` 里的账号登录（默认演示账号 `admin / runos-demo`）。

## 为什么选 RunOS

- **数据进来就能用**：导入、浏览、筛选、合并、拆分、格式转换、版本快照，数据集先规整，再谈模型。
- **标注不用纯手点**：人工标注打底，AI 辅助、大模型自动标注跟上，复核通过再应用——最磨人的环节先让模型跑一遍。
- **算力不用守着一台机器**：手头几台服务器全注册为算力节点，终端分发任务、节点跑训练、结果回传终端统一看。
- **不会训模型也能先检测**：一句话描述目标，大模型直接给结论、给告警、给证据。
- **不用拼七八个工具**：数据集、标注、训练、推理全在一个服务里，自托管部署，开箱即用。

## 重点功能

### 算力节点

一台服务器做终端（控制端），其他服务器做算力：RunOS 装在终端上统一控制，训练任务从终端分发到各算力节点，跑完结果再回传到终端——显卡在哪，算力就在哪，不用挨个 SSH 上去折腾：

- 连接方式为 SSH 或 API；节点信息包括主机、端口、账号、密码或密钥认证、工作目录与 Python 环境（system / conda / venv）。
- 添加节点后自动检测：连通性、操作系统、工作目录、GPU 列表，检测报告含错误信息；首次连接需确认主机指纹。
- 内网服务器可经公网跳板机转发（本机 → 跳板机 → 内网服务器），跳板机同样支持密码或密钥认证。
- 算力节点凭据操作仅允许本机 RunOS 访问。

![算力节点](docs/screenshots/compute-nodes.png)

### 大模型自动标注

标注最磨人的部分，先让大模型跑一遍，人只负责把关：

- 模型接入：通义千问 Qwen-VL（阿里云 DashScope，国内直连）、OpenAI GPT-4o、本地 vLLM、本地 Ollama，均为 OpenAI 兼容接口；另有 VLX-Seek 专用配置。
- 流程：配置接口与类别 → 下发自动标注任务 → 任务可暂停、恢复、停止 → 逐条复核标注结果 → 应用到数据集。
- 标注结果、任务日志与原始输出均可查看，低成本替代纯人工标注。

![自动标注](docs/screenshots/auto-annotation.png)

### 大模型检测

想找水面上的船，不用先标几百张船、训一个模型——直接问一句：

- 例如“寻找水面上的船只”，由云端 VLX-Seek 检测器返回结论，并按规则告警。
- 任务附带结果 JSON、结果视频、快照与告警证据（告警证据可恢复查看）；另有视频任务、逐帧与单图推理。
- 本地 YOLOE 模型状态、VLX-Seek 算力节点连通性均可在页面查看；未配置时页面会明确提示。

![大模型检测](docs/screenshots/lmm.png)

## 功能

- **数据集管理**：导入、浏览、筛选、合并、拆分、格式转换、版本快照
- **智能标注**：人工标注、AI 辅助标注、大模型自动标注（VLM 多模态识别），人工复核流程
- **模型与训练**：YOLO / RF-DETR 等训练任务下发、进度跟踪、评测对比
- **算力节点**：注册多台服务器作为算力节点，训练与推理任务跨节点调度
- **实时多模态识别**：视频与图像流的多模态大模型识别与告警证据留存
- **相机与视频**：视频抽帧、切片预览、平铺（tiling）处理

## 界面巡览

登录页与全部视图的截图及功能说明见 [docs/pages-tour.md](docs/pages-tour.md)（空数据环境下截取，需要数据集的页面显示“未选择数据集”占位；英文版见 [docs/pages-tour.en.md](docs/pages-tour.en.md)）。

![登录页](docs/screenshots/login.png)

![首页](docs/screenshots/home.png)

## 架构

```
frontend/   Next.js 前端（app/、components/、lib/）
backend/    FastAPI 后端（app/，uvicorn :8010）
            routers/：标注、数据集、训练、算力节点、多模态、推理、评测……
            compute/：节点调度与远程任务
            lmm/、vlm_annotator：多模态大模型能力
```

前端通过服务端代理访问后端，浏览器不直连后端；数据默认存在 Docker 卷 `runos-data`（后端 `RUNOS_DATA_ROOT=/data`）。

## 登录方式

- **local（默认）**：独立部署的本地单管理员登录。账号密码来自 `.env` 的
  `RUNOS_ADMIN_USER` / `RUNOS_ADMIN_PASSWORD`；会话为 HMAC 签名 token，
  签名密钥首次启动时生成并保存在数据卷（`.session_secret`），重启不失效。

## 配置

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `RUNOS_ADMIN_USER` / `RUNOS_ADMIN_PASSWORD` | 本地管理员账号 | `admin` / `runos-demo` |
| `RUNOS_PORT` | 前端发布端口 | `19000` |
| `RUNOS_BIND` | 前端绑定地址（默认只绑本机回环；局域网访问时填本机局域网 IP） | `127.0.0.1` |
| `RUNOS_DATA_ROOT` | 后端数据目录（容器内） | `/data` |

## 说明与限制

- 本仓库不含任何真实密钥、模型权重与业务数据；模型权重在运行时按需下载，
  大模型识别需要使用者自备模型服务或 API 密钥。
- 后端镜像较大（PyTorch + Ultralytics 体系），首次构建需要一些时间；
  有 GPU 的机器可把后端换成 GPU 基础镜像以启用 CUDA。
- 训练任务本身需要算力与模型权重；一键部署验证的是服务链路，不是训练效果。
- 许可证：MIT，版权人 RunZe Huang。
