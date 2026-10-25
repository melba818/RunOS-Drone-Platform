[English](pages-tour.en.md) | 简体中文

# RunOS 界面巡览

截图环境：空数据（无数据集、无已下载模型、无算力节点），登录后逐页截取。
需要数据集的页面在空环境下会显示“未选择数据集”占位，这是正常空状态，不是故障。
登录页已改为与应用内页一致的浅色风格。

## 登录页

本地管理员登录入口。默认演示账号 `admin / runos-demo`，正式暴露部署前请在 `.env` 中修改密码。

![登录页](screenshots/login.png)

## 首页

默认打开数据集视图（与“数据集”页同一视图）。

![首页](screenshots/home.png)

## 数据集

加载并管理图像集合：上传 ZIP、浏览本地文件夹；空环境显示上传与载入入口。

![数据集](screenshots/datasets.png)

## 数据看板

查看当前数据集的统计信息、类别分布与分析洞察；未选数据集时显示占位提示。

![数据看板](screenshots/dashboard.png)

## 图库

以可视化网格浏览全部图像，点击图像可在标注器中打开；未选数据集时显示占位提示。

![图库](screenshots/gallery.png)

## 筛选排序

逐张查看图像，用键盘快捷键快速整理数据；未选数据集时显示占位提示。

![筛选排序](screenshots/sorting.png)

## 重复检测

自动查找并移除数据集中重复或高度相似的图像；未选数据集时显示占位提示。

![重复检测](screenshots/duplicate-detection.png)

## 标注

在图像上绘制边界框与标签，教 AI 识别目标对象；未选数据集时显示占位提示。

![标注](screenshots/annotate.png)

## 类别

定义 AI 需要学习的目标类别（例如 car、person）；未选数据集时显示占位提示。

![类别](screenshots/classes.png)

## 算力节点（重点）

把多台服务器注册为算力节点，训练、推理与云端大模型都跑在这些节点上，是 RunOS 跨服务器调配算力的入口。

- 添加节点：连接方式为 SSH 或 API，填主机、端口、账号、密码或密钥认证、工作目录与 Python 环境（system / conda / venv）。
- 自动检测：添加后检测连通性、操作系统、工作目录与 GPU 列表，报告含错误信息；首次连接需确认主机指纹。
- 跳板机转发：内网服务器可经公网跳板机转发（本机 → 跳板机 → 内网服务器），跳板机同样支持密码或密钥认证。
- 凭据安全：算力节点凭据操作仅允许本机 RunOS 访问。

当前节点列表为空，截图为添加节点入口。

![算力节点](screenshots/compute-nodes.png)

## 大模型配置 / 大模型自动标注（重点）

接入视觉大模型，用大模型给数据自动打标签，人工复核后再应用到数据集，低成本替代纯人工标注。

- 模型接入：通义千问 Qwen-VL（阿里云 DashScope，国内直连）、OpenAI GPT-4o、本地 vLLM、本地 Ollama，均为 OpenAI 兼容接口；另有 VLX-Seek 专用配置。
- 流程：配置接口与类别 → 下发自动标注任务 → 任务可暂停、恢复、停止 → 逐条复核标注结果 → 应用到数据集。
- 结果可查：标注结果、任务日志与原始输出均可查看。

截图中已有一条 VLX-Seek 专用配置（模型 VLX-Seek-1.5-10B），并可新建 OpenAI 兼容模型。

![大模型配置](screenshots/vlm.png)

## 多模态目标检测（重点）

用自然语言描述目标，让大模型直接给出检测结论并告警，是 RunOS 的实时多模态识别入口。

- 描述即检测：例如“寻找水面上的船只”，由云端 VLX-Seek 检测器返回结论，并按规则告警。
- 证据留存：任务附带结果 JSON、结果视频、快照与告警证据（告警证据可恢复查看）；另有视频任务、逐帧与单图推理。
- 状态可见：本地 YOLOE 模型状态、VLX-Seek 算力节点连通性均可在页面查看；未配置时页面会明确提示。

当前未配置 VLX-Seek 算力节点，截图为任务与配置入口。

![多模态目标检测](screenshots/lmm.png)

## 训练 / 验证 / 测试

将数据集划分为训练集、验证集与测试集，以便正确评估模型；未选数据集时显示占位提示。

![训练 / 验证 / 测试](screenshots/split.png)

## 数据增强

自动应用翻转、旋转与颜色变化来扩充训练数据；未选数据集时显示占位提示。

![数据增强](screenshots/augmentation.png)

## 视频抽帧

从视频文件中提取帧图像，用作训练素材。

![视频抽帧](screenshots/video-extraction.png)

## 格式转换

在 YOLO、COCO、Pascal VOC 等格式之间转换数据集。

![格式转换](screenshots/convert.png)

## 合并

将两个或多个独立数据集合并为一个统一数据集。

![合并](screenshots/merge.png)

## YAML 配置

生成启动模型训练所需的配置文件；未选数据集时显示占位提示。

![YAML 配置](screenshots/yaml-wizard.png)

## 模型训练

在已标注的数据集上训练模型，实时跟踪 loss 与精度变化。截图为训练配置入口。

![模型训练](screenshots/training.png)

## 推理

在图像、视频文件或实时摄像头上测试已训练模型，支持目标跟踪与 ReID。

![推理](screenshots/inference.png)

## 评估

多模型横向评估，对比 mAP、precision、recall 等指标。

![评估](screenshots/evaluate.png)

## 模型

查看与管理已训练的模型，可运行推理测试其效果；当前模型目录为空（yolov8n 未下载）。

![模型](screenshots/models.png)

## 批量任务

监控并管理导出、增强等长时间运行的后台任务；当前无任务。

![批量任务](screenshots/batch-jobs.png)

## 健康检查

扫描数据集的质量问题，例如缺失标签或损坏的图像；未选数据集时显示占位提示。

![健康检查](screenshots/health.png)

## 对比

对两个或多个数据集进行并排的统计对比。

![对比](screenshots/compare.png)

## 快照

随时保存数据集快照，需要时可恢复；未选数据集时显示占位提示。

![快照](screenshots/snapshots.png)

## 设置

配置应用偏好与后端连接（API URL、存储路径、GPU、偏好设置）。
注：后端状态徽标显示“未连接”，是因为已保存的 API URL 为空（表单中的 `/api/proxy` 为草稿默认值）；
应用数据实际走 `/api/proxy/api/*` 代理，实测连通。

![设置](screenshots/settings.png)
