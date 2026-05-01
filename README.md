# FlameGuardWeb

FlameGuardWeb 是一个面向现场工作人员的预热炉监视 Web 应用，用于观察炉内运行状态、垃圾组分变化、温度趋势和控制建议。应用提供大屏式界面，可在本机或局域网中通过浏览器访问。

> 当前版本以实时仿真监控为主，适合演示、调试和现场界面验证。接入真实设备前，请结合现场安全规程、通信联锁和工程验收流程使用。

## 功能概览

- **炉内状态监视**：展示炉温、烟囱温度、烟气速度、预热炉出口物料状态等关键指标。
- **垃圾组分观察**：支持菜叶、西瓜皮、橙子皮、肉、杂项混合、米饭等组分比例输入与展示。
- **实时趋势图**：连续刷新温度、控制指令和扰动趋势，便于观察运行变化。
- **控制建议展示**：显示控制器建议值、实际执行值、安全裕度和运行状态。
- **三维模型预览**：保留炉体三维模型视图，用于现场大屏展示。
- **局域网访问**：可部署在一台电脑或工控机上，让同一局域网内的多台设备访问。

## 快速开始

推荐使用 Docker Compose 启动。

### 1. 安装 Docker

请先安装 Docker Desktop 或 Docker Engine，并确认命令可用：

```bash
docker --version
docker compose version
```

### 2. 准备配置

进入项目目录：

```bash
cd FlameGuardWeb
```

复制配置模板：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

### 3. 启动应用

前台启动：

```bash
docker compose up --build
```

后台启动：

```bash
docker compose up --build -d
```

启动完成后，在浏览器打开：

```text
http://127.0.0.1:5000
```

停止服务：

```bash
docker compose down
```

## 基本使用

打开页面后，可以看到三栏监视界面：

1. **左侧面板**：查看系统状态，输入或校正垃圾组分。
2. **中间面板**：查看三维模型和实时趋势图。
3. **右侧面板**：查看组分占比、控制建议、执行状态和安全状态。

常用操作：

- 点击 **启动监控** 开始刷新运行数据。
- 点击 **暂停监控** 暂停实时刷新。
- 修改垃圾组分后，点击 **提交组分 / 刷新监控** 更新当前观测。
- 点击 **复位仿真** 恢复到初始运行状态。

## 配置说明

`.env.example` 提供了常用配置模板，复制成 `.env` 后可按需修改：

```env
FLAMEGUARD_PORT=5000
FLAMEGUARD_ENV=production
TZ=Asia/Shanghai
FLAMEGUARD_REFRESH_MS=250
FLAMEGUARD_HISTORY_LIMIT=1800
FLAMEGUARD_DASHBOARD_DEFAULT_LIMIT=360
FLAMEGUARD_DASHBOARD_MAX_LIMIT=1200
FLAMEGUARD_HEALTH_REQUIRE_NMPC=1
```

常见配置项：

| 配置 | 说明 | 建议值 |
|---|---|---|
| `FLAMEGUARD_PORT` | 宿主机访问端口 | `5000` |
| `FLAMEGUARD_ENV` | 运行环境标识 | `production` |
| `TZ` | 容器时区 | `Asia/Shanghai` |
| `FLAMEGUARD_REFRESH_MS` | 前端刷新间隔，单位 ms | 现场演示 `200`-`250`，普通电脑 `300`-`500` |
| `FLAMEGUARD_HISTORY_LIMIT` | 后端保留的历史点数量 | `1800` |
| `FLAMEGUARD_DASHBOARD_DEFAULT_LIMIT` | dashboard 默认返回历史点数量 | `360` |
| `FLAMEGUARD_DASHBOARD_MAX_LIMIT` | dashboard 最大返回历史点数量 | `1200` |
| `FLAMEGUARD_HEALTH_REQUIRE_NMPC` | 健康检查是否要求实时 NMPC 成功初始化 | 生产环境 `1` |

修改 `.env` 后重启：

```bash
docker compose up --build -d
```

## Docker Compose 部署说明

当前容器内运行的是 Flask + Gunicorn，并保持 **1 个 worker + 多线程**：

```text
workers = 1
threads = 4
```

原因是实时 plant、快速 NMPC 和 telemetry store 当前放在单个 Python 进程内存中。多个 worker 会形成多个相互独立的仿真状态，导致页面看到的数据不一致。后续如果把 NMPC solver 和 telemetry 存储拆成独立服务，再考虑横向扩展。

项目中的 Docker 相关文件：

```text
Dockerfile                 容器镜像构建文件
docker-compose.yml         本地构建并启动服务
.dockerignore              构建镜像时忽略无关文件
.env.example               环境变量模板
.github/workflows/         GitHub Actions 自动构建与发布配置
logs/                      预留日志目录，compose 会挂载到容器 /app/logs
```

常用命令：

```bash
# 构建并后台启动
docker compose up --build -d

# 查看状态
docker compose ps

# 查看日志
docker compose logs -f flameguardweb

# 重新构建，不使用缓存
docker compose build --no-cache

# 停止并删除容器网络
docker compose down
```

## 局域网访问

容器默认把服务映射到宿主机：

```text
0.0.0.0:${FLAMEGUARD_PORT:-5000} -> container:5000
```

同一局域网内其他电脑可以访问：

```text
http://宿主机IP:5000
```

如果无法访问，请检查系统防火墙是否允许 Docker 或 5000 端口入站。

如果 5000 端口被占用，编辑 `.env`：

```env
FLAMEGUARD_PORT=8080
```

然后重启并访问：

```text
http://127.0.0.1:8080
```

## 健康检查

容器内置健康检查会访问：

```text
http://127.0.0.1:5000/healthz
```

也可以手动检查：

```bash
curl -i http://127.0.0.1:5000/healthz
```

正常情况下会返回 HTTP 200，并包含类似内容：

```json
{
  "ok": true,
  "service": "FlameGuardWeb",
  "status": "ready",
  "mode": "realtime_fast_nmpc",
  "expected_mode": "realtime_fast_nmpc",
  "checks": {
    "runtime_import_ok": true,
    "runtime_initialized": true,
    "startup_ok": true,
    "mode_ok": true,
    "dashboard_snapshot_ok": true,
    "realtime_nmpc_required": true
  }
}
```

生产环境默认要求实时 NMPC 初始化成功。如果初始化失败并降级到 `phase1_fallback`，`/healthz` 会返回 HTTP 503，`docker compose ps` 中容器会显示为 `unhealthy`。

临时演示时如果明确允许 fallback，可以在 `.env` 中设置：

```env
FLAMEGUARD_HEALTH_REQUIRE_NMPC=0
```

然后重启：

```bash
docker compose up --build -d
```

## GitHub Actions 自动发布 Docker 镜像

仓库已包含：

```text
.github/workflows/docker-publish.yml
```

该 workflow 会自动完成 Docker 构建与 GHCR 发布：

```text
Pull Request -> 只构建镜像，不推送
main 分支    -> 构建并推送 latest / main / sha-* 标签
v*.*.* tag   -> 构建并推送版本标签，例如 0.2.0
手动运行     -> 可在 Actions 页面 workflow_dispatch 触发
```

默认镜像名：

```text
ghcr.io/harmonese/flameguardweb
```

发布一个版本镜像：

```bash
git tag v0.2.0
git push origin v0.2.0
```

如果 GHCR package 是 private，目标机器需要先登录：

```bash
echo YOUR_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

## 使用 GHCR 镜像部署

本仓库默认的 `docker-compose.yml` 是本地构建模式：

```yaml
build:
  context: .
  dockerfile: Dockerfile
image: flameguardweb:0.2.0
```

如果目标机器不需要源码构建，而是直接使用 GitHub Actions 发布的镜像，可以把 compose 服务改成：

```yaml
services:
  flameguardweb:
    image: ghcr.io/harmonese/flameguardweb:latest
    container_name: flameguardweb
    ports:
      - "${FLAMEGUARD_PORT:-5000}:5000"
    environment:
      TZ: ${TZ:-Asia/Shanghai}
      FLAMEGUARD_ENV: ${FLAMEGUARD_ENV:-production}
      FLAMEGUARD_HISTORY_LIMIT: ${FLAMEGUARD_HISTORY_LIMIT:-1800}
      FLAMEGUARD_DASHBOARD_DEFAULT_LIMIT: ${FLAMEGUARD_DASHBOARD_DEFAULT_LIMIT:-360}
      FLAMEGUARD_DASHBOARD_MAX_LIMIT: ${FLAMEGUARD_DASHBOARD_MAX_LIMIT:-1200}
      FLAMEGUARD_REFRESH_MS: ${FLAMEGUARD_REFRESH_MS:-250}
      FLAMEGUARD_HEALTH_REQUIRE_NMPC: ${FLAMEGUARD_HEALTH_REQUIRE_NMPC:-1}
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

然后执行：

```bash
docker compose pull
docker compose up -d
```

## Docker Compose 环境测试

本地或服务器上可以按下面顺序验证：

```bash
cp .env.example .env

docker compose down
docker compose build --no-cache
docker compose up -d

docker compose ps
docker compose logs -f flameguardweb
```

接口测试：

```bash
curl -i http://127.0.0.1:5000/healthz
curl http://127.0.0.1:5000/api/config
curl http://127.0.0.1:5000/api/dashboard
```

页面测试：

```text
http://127.0.0.1:5000
```

期望结果：

- `docker compose ps` 显示容器 `Up`，并最终变为 `healthy`。
- `/healthz` 返回 HTTP 200。
- `/api/config` 返回刷新间隔、历史长度等运行配置。
- `/api/dashboard` 返回实时 dashboard 数据。
- 页面可以正常启动、暂停、提交组分和复位仿真。

如果 `/healthz` 返回 HTTP 503，查看返回体里的 `startup_error`、`snapshot_error` 和 `checks` 字段，再查看容器日志：

```bash
docker compose logs -f flameguardweb
```

## 常见问题

### 页面打不开怎么办？

先看容器是否运行：

```bash
docker compose ps
```

再看日志：

```bash
docker compose logs -f flameguardweb
```

确认访问地址：

```text
http://127.0.0.1:5000
```

如果是局域网访问，确认宿主机 IP 和防火墙规则。

### 页面刷新太慢或太快怎么办？

修改 `.env`：

```env
FLAMEGUARD_REFRESH_MS=250
```

推荐范围：

- 演示大屏：`200` 到 `250`
- 普通电脑：`300` 到 `500`
- 高性能设备：可适当降低到 `100` 左右

刷新越快，浏览器和后端 CPU 压力越大。

### Docker 构建很慢怎么办？

第一次构建需要下载 Python 基础镜像并安装 NumPy/SciPy 等依赖。后续只要 `requirements.txt` 不变，Docker 会复用缓存。GitHub Actions 中也配置了 BuildKit cache。

### 这个应用可以直接控制现场设备吗？

当前版本主要用于监视、演示和仿真验证。接入真实设备前，需要经过现场通信、权限、安全联锁和工程验收配置，不建议把本应用作为唯一安全依据。

## 项目状态

当前版本定位为：

```text
预热炉现场监视界面 + 实时仿真观测 + 控制建议展示
```

正式交付时建议发布版本包，例如：

```text
FlameGuardWeb-v0.2.0-docker.zip
```

包内应包含完整工程、`Dockerfile`、`docker-compose.yml`、`.env.example` 和本 README。
