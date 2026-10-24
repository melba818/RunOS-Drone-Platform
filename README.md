# RunOS

RunOS 是一个面向无人机与视觉数据的智能数据引擎：数据集管理、智能标注、模型训练与评测、算力节点调度，全部跑在一个可独立部署的服务里。

它由 visos 演进而来：保留了 visos 的数据与标注底座，加入了实时的多模态大模型识别、大模型自动标注，并对标大疆司空 2（DJI FlightHub 2）的使用体验；同时内置算力节点，可以把不同服务器的算力统一调配起来做算法训练。

## 功能

- **数据集管理**：导入、浏览、筛选、合并、拆分、格式转换、版本快照
- **智能标注**：人工标注、AI 辅助标注、大模型自动标注（VLM 多模态识别），人工复核流程
- **模型与训练**：YOLO / RF-DETR 等训练任务下发、进度跟踪、评测对比
- **算力节点**：注册多台服务器作为算力节点，训练与推理任务跨节点调度
- **实时多模态识别**：视频与图像流的多模态大模型识别与告警证据留存
- **相机与视频**：视频抽帧、切片预览、平铺（tiling）处理

## 快速开始

```bash
cp .env.example .env   # 先改掉演示密码
docker compose up -d --build
```

打开 http://127.0.0.1:19000 ，用 `.env` 里的账号登录（默认演示账号 `admin / runos-demo`）。

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
- **nexushive**：对接 RunHive 门户票据登录（`RUNOS_AUTH_MODE=nexushive`），
  仅在 RunOS 作为 RunHive 一部分运行时使用。

## 配置

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `RUNOS_AUTH_MODE` | `local` 或 `nexushive` | `local` |
| `RUNOS_ADMIN_USER` / `RUNOS_ADMIN_PASSWORD` | 本地管理员账号 | `admin` / `runos-demo` |
| `RUNOS_PORT` | 前端发布端口（仅本机回环） | `19000` |
| `RUNOS_DATA_ROOT` | 后端数据目录（容器内） | `/data` |

## 说明与限制

- 本仓库不含任何真实密钥、模型权重与业务数据；模型权重在运行时按需下载，
  大模型识别需要使用者自备模型服务或 API 密钥。
- 后端镜像较大（PyTorch + Ultralytics 体系），首次构建需要一些时间；
  有 GPU 的机器可把后端换成 GPU 基础镜像以启用 CUDA。
- 训练任务本身需要算力与模型权重；一键部署验证的是服务链路，不是训练效果。
- 许可证：MIT，版权占位为 `RunOS Authors`，正式发布前可再定法律主体名。

## English summary

RunOS is a standalone intelligent data engine for drone and vision data:
dataset management, smart annotation (including LLM-powered auto-annotation),
model training and evaluation, and compute-node orchestration across servers.
It evolved from visos, adds real-time multimodal LLM recognition, and targets
a DJI FlightHub 2-class user experience. `docker compose up -d --build`,
then open http://127.0.0.1:19000 (demo account `admin / runos-demo`).
