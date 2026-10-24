"""model_runtime —— 多算力节点 / 多模型部署的最小运行时契约层。

P0/P1 范围（本轮）：
  * target.DetectionTarget          —— 统一检测目标（label 才允许进 data.yaml）
  * model_catalog.ModelCatalog      —— 只读 Model 注册表（内置 + Legacy Bridge）
  * deployment_store.DeploymentStore—— Deployment 静态配置的持久化抽象
  * registry.DeploymentRegistry     —— Deployment 静态配置 CRUD（不含运行状态）
  * runtime_registry.DeploymentRuntimeRegistry —— 运行状态（仅内存）
  * resolver.DeploymentResolver     —— 纯候选选择（不推理、不改状态、不 failover）
  * options.resolve_options         —— request > deployment override > model default
  * legacy_bridge                   —— 旧 VLX / VLM 配置的只读映射
  * secrets.LegacySecretResolver    —— auth_ref 前缀解析（只读）

硬约束（P0/P1）：
  1. FEATURE_MODEL_RUNTIME_V1 必须为 0，本包不得进入任何真实推理链路。
  2. 本包不得 import 全局 settings / service 单例，依赖一律注入。
  3. 不得修改 lmm/target_spec.py；不得复制 api_key / token 到 Model / Deployment。
  4. load() 不得自动创建 deployments.json。
"""
