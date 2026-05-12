# Hrio-Commerce v7.13.0 发布说明

## 本版更新

- 普通分类新增：`🍌 普通单图生成`
- 普通分类新增：`🍌 普通生视频（单输出）`
- 保留：`🍌 Banana｜普通三视图并发`
- 保留：`🍌 Banana｜生视频`
- 修复：冬之韵模板面板打不开
- 修复：面板路由重复注册导致 ComfyUI 重载后打不开
- 视频节点会把远程视频下载到 ComfyUI temp 目录，并通过节点 UI 预览

## 发布到 ComfyUI 插件管理器

1. 解压本包。
2. 把全部文件覆盖到你的 GitHub 仓库 `moki-flank/Hrio-Commerce` 根目录。
3. GitHub 仓库进入：Settings → Secrets and variables → Actions → New repository secret。
4. 新建 secret：`REGISTRY_ACCESS_TOKEN`。
5. secret 值填 Comfy Registry 发布 API Key。
6. 提交并推送到 `main` 或 `master`。
7. 因为 `pyproject.toml` 版本号已升级到 `7.13.0`，GitHub Actions 会自动执行 `.github/workflows/publish_action.yml`。
8. Actions 成功后，Comfy Registry / ComfyUI-Manager 会识别为新版。

## 注意

- 每次发布新版都必须修改 `pyproject.toml` 的 `[project].version`。
- 版本号必须是三段语义版本号，例如 `7.13.0`，不要使用 `7.13.0-beta` 这类后缀。
- 如果 Manager 里仍显示旧版本，先在 ComfyUI-Manager 里刷新节点数据库，再重启 ComfyUI。
