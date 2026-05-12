Hrio-Commerce modified v7.13.0

本版修复：
1. 冬之韵模板面板不再 window.open 新页面，而是在当前 ComfyUI 页面中以内嵌 iframe 弹窗方式打开。
2. 重复点击“打开冬之韵模板面板”不会重复创建页面，只会复用同一个弹窗。
3. 节点同步改为默认自动启用：ComfyUI 启动、节点创建、打开面板时都会自动从 /banana/ecommerce-prompt-config 同步配置并美化节点。
4. 浮窗继续保留四个入口：打开冬之韵模板面板、自动化、清除自动化、历史记录。
5. 保留 v7.12 的自动化清除与历史缓存功能。

替换方式：关闭 ComfyUI -> 删除 custom_nodes/Hrio-Commerce -> 解压新版 Hrio-Commerce 到 custom_nodes -> 启动 ComfyUI -> Ctrl+F5 强刷。
