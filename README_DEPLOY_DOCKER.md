# FlameGuardWeb Docker Compose 部署说明

本说明用于把 FlameGuardWeb 作为容器服务发布。当前容器内运行的是 Flask + Gunicorn，保持 **1 个 worker + 多线程**，原因是实时仿真状态和 telemetry 缓存在 Web 进程内存中；如果开多个 worker，每个 worker 会拥有不同的仿真状态。

## 1. 前置条件

安装 Docker Desktop 或 Docker Engine，并确认命令可用：

```bash
docker --version
docker compose version
```

## 2. 文件说明

```text
Dockerfile             容器镜像构建文件
docker-compose.yml     一键启动服务
.dockerignore          构建镜像时忽略的文件
.env.example           可选环境配置模板
logs/                  预留日志目录，compose 会挂载到容器 /app/logs
```

## 3. 最快启动

在 `FlameGuardWeb` 目录下执行：

```bash
docker compose up --build
```

启动成功后打开：

```text
http://127.0.0.1:5000
```

## 4. 后台运行

```bash
docker compose up --build -d
```

查看状态：

```bash
docker compose ps
```

查看日志：

```bash
docker compose logs -f flameguardweb
```

停止服务：

```bash
docker compose down
```

## 5. 修改端口、刷新速度、历史长度

复制配置模板：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```env
FLAMEGUARD_PORT=5000
FLAMEGUARD_REFRESH_MS=250
FLAMEGUARD_HISTORY_LIMIT=1800
FLAMEGUARD_DASHBOARD_DEFAULT_LIMIT=360
FLAMEGUARD_DASHBOARD_MAX_LIMIT=1200
TZ=Asia/Shanghai
```

改完后重启：

```bash
docker compose up --build -d
```

如果要让前端刷新更快，例如 100 ms：

```env
FLAMEGUARD_REFRESH_MS=100
```

注意：刷新越快，浏览器与后端 CPU 压力越大。现场演示推荐 200–250 ms；性能较弱的电脑推荐 500 ms。

## 6. 局域网访问

容器默认把服务映射到宿主机：

```text
0.0.0.0:5000 -> container:5000
```

同一局域网内其他电脑可以访问：

```text
http://宿主机IP:5000
```

如果无法访问，检查系统防火墙是否允许 Docker/5000 端口入站。

## 7. 健康检查

Compose 会使用容器内健康检查访问：

```text
http://127.0.0.1:5000/healthz
```

也可以手动检查：

```bash
curl http://127.0.0.1:5000/healthz
```

返回类似：

```json
{"ok": true, "service": "FlameGuardWeb"}
```

## 8. 常见问题

### 端口被占用

如果 5000 被占用，编辑 `.env`：

```env
FLAMEGUARD_PORT=8080
```

然后访问：

```text
http://127.0.0.1:8080
```

### 为什么 Gunicorn 只开 1 个 worker？

FlameGuardWeb 当前把实时 plant、快速 NMPC、telemetry store 放在单个 Python 进程内存中。多个 worker 会形成多个相互独立的仿真世界，导致页面看到的状态不一致。后续如果把 NMPC solver 和 telemetry 存储拆成独立服务，再考虑横向扩展。

### Docker 构建很慢

第一次构建需要下载 Python 基础镜像和安装 NumPy/SciPy。后续只要 `requirements.txt` 不变，Docker 会复用缓存。

## 9. 发布建议

正式交付时建议发布：

```text
FlameGuardWeb-v0.2.0-docker.zip
```

里面包含完整工程、`Dockerfile`、`docker-compose.yml`、`.env.example` 和本部署说明。现场人员只需要安装 Docker 后执行：

```bash
docker compose up -d
```
