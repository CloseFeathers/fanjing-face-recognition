# 心流元素人脸识别系统 — 优化与修复计划

> 目标：使项目达到开源发布标准，确保安全、稳定、可维护。  
> 审计日期：2026-04-14  
> 交叉验证日期：2026-04-14（已逐项读源码确认每个问题真实存在）  
> 当前评分：6.8/10 → 目标：9.0/10

---

## 目录

- [Phase 1: 安全漏洞修复（Critical）](#phase-1-安全漏洞修复critical)
- [Phase 2: 运行时缺陷修复（High）](#phase-2-运行时缺陷修复high)
- [Phase 3: 开源基础设施](#phase-3-开源基础设施)
- [Phase 4: 代码质量提升](#phase-4-代码质量提升)
- [Phase 5: 架构重构](#phase-5-架构重构)
- [Phase 6: 工程化与 CI/CD](#phase-6-工程化与cicd)
- [附录 A: 完整问题清单](#附录-a-完整问题清单)
- [附录 B: 文件影响矩阵](#附录-b-文件影响矩阵)

---

## 交叉验证修订记录

> 以下为初版计划经逐项源码验证后的修订，确保每个问题真实存在、方案切实可行。

| 修订项 | 原计划内容 | 验证发现 | 修订结果 |
|--------|-----------|----------|----------|
| **1.2 视频路径** | 硬编码限制 `uploads/` 目录 | 会阻断合法的本地视频调试场景 | 改为可配置的允许目录列表（环境变量 `ALLOWED_VIDEO_DIRS`） |
| **1.3 模型路径** | 单文件名白名单 | 不支持用户添加新模型，过于死板 | 改为目录约束（`models/` 下）+ 扩展名白名单 |
| **1.4 XSS (index.html)** | 仅建议 `escapeHtml()` | `onclick` 中存在 JS 字符串上下文，纯 HTML 转义可被 `');alert(1);//` 绕过 | 必须改用 `addEventListener` + `textContent` 双重防护 |
| **1.4 XSS (index_v2.html)** | 未提及 avatar 字段 | `p.avatar` base64 拼入 `src="..."` 可能逃逸属性引号 | 新增 avatar base64 正则白名单校验 |
| **1.5 pickle** | 建议改用 `joblib` + SHA256 | `joblib.load` 内部仍用 pickle，**不解决根本问题**；路径不可通过 API 控制 | 改为 XGBoost 原生 JSON 格式 + 纯 JSON 元数据，彻底消除 pickle；风险降为 🟠Medium |
| **2.2 线程竞态** | 对每次读写加 `state.lock` | 30fps 下加锁可能成为性能瓶颈；`state.lock` 虽已定义但从未使用；GIL 保护单操作 | 改为原子快照替换模式（后台线程构建快照，原子赋值替换引用） |
| **2.3 STrack ID** | 标记为 🟠High，建议加锁 | 验证 `tracker.step()` 仅在主线程调用，IdentityWorker/MouthWorker 不触碰 STrack | 降为 🔵Info，当前无需修复，仅作防御性建议 |

**第二轮验证修订（2026-04-14）**:

| 修订项 | 原计划内容 | 验证发现 | 修订结果 |
|--------|-----------|----------|----------|
| **1.5 pickle → XGBoost JSON** | 仅描述 `Booster.load_model` | 模型类型为 `XGBClassifier`（sklearn API），推理用 `predict_proba`；需改为 `Booster.predict(DMatrix)` | 补充完整推理链路修改方案，确认二分类 logistic 下输出兼容 |
| **1.6 认证 /video_feed** | 未说明 img 标签无法携带 Header | `<img src="/video_feed">` 无法加 X-API-Key header | 新增签名 URL 方案（HMAC + 时效），或绑定 127.0.0.1 |
| **2.2 线程快照** | 仅覆盖 `track_to_person` 和 `track_similarities` | `person_identity_states`、`person_to_registered` 也被跨线程读写 | 快照扩展为 4 个字典的统一快照对象 |
| **新增 2.4** | 不存在 | `/api/log`、`/api/embed_log` 的 `int(n)` 无 try/except | 新增：查询参数类型转换捕获 |
| **新增 2.5** | 不存在 | `source_id` 含 `file:/Users/xxx/video.mp4` 绝对路径 | 新增：API 响应中的路径信息泄露 |
| **新增 2.6** | 不存在 | `save()` 先写 npz 后写 json，非原子操作 | 新增：RegisteredPersonDB 原子写入 |
| **新增 2.10** | 不存在 | `/video_feed` 无连接数限制，可 DoS | 新增：并发流限制 |

**第三轮验证修订（2026-04-14）— 修复方案自身缺陷检查**:

| 修订项 | 原方案内容 | 发现的问题 | 修订结果 |
|--------|-----------|-----------|----------|
| **2.10 并发计数器** | `_active_streams += 1`（全局变量直接递增） | `+= 1` 是 LOAD+ADD+STORE 三条字节码，多线程下非原子 → 计数器本身有竞态 | 加 `threading.Lock` 保护计数器读写 |
| **2.2 快照初始化** | 仅在 `_alignment_step` 末尾赋值 `_identity_snapshot` | `_generate_frames` 首帧时 `_alignment_step` 尚未执行，`_identity_snapshot` 不存在 → `AttributeError` | 在 `PipelineState.__init__` 中初始化为空快照 |
| **1.6 签名URL** | `int(ts)` 无异常捕获 | 非数字 `ts` 参数触发 `ValueError` → 500 | 添加 `try/except` |
| **1.1 上传响应** | 返回 `str(save_path)` 其中 save_path 已 `.resolve()` | resolve 后为绝对路径（如 `/Users/.../uploads/xxx.mp4`），泄露服务器文件系统结构，与 2.5 修复矛盾 | 改为返回相对路径 `uploads/xxx.mp4` |
| **1.6 Jinja2 注入** | `const API_KEY = '{{ api_key }}'` | 若 key 含单引号或反斜杠会破坏 JS 语法，甚至构成 XSS | 改用 `{{ api_key \| tojson }}`（自动加引号并转义） |
| **2.8 CSP** | `script-src 'unsafe-inline'` | unsafe-inline 允许所有内联脚本，削弱 XSS 防护；但当前 HTML 大量内联脚本无法立即消除 | 保留 unsafe-inline 并注释说明原因；补充 `object-src 'none'`、`base-uri 'self'`、`form-action 'self'` 增强其他维度防护 |

**第四轮验证修订（2026-04-14）— 方案间交叉一致性检查**:

| 修订项 | 原方案内容 | 发现的问题 | 修订结果 |
|--------|-----------|-----------|----------|
| **1.3 模型路径错误信息** | `raise ValueError(f"模型路径必须在 {MODELS_DIR} 下")` | `MODELS_DIR` 是 resolve 后的绝对路径，异常被 `api_start` 的 `except` 捕获后 `str(e)` 返回客户端，**泄露服务器文件系统结构**——与 2.5/2.7 修复目标矛盾 | 错误信息改为不含路径的泛化描述 |
| **1.1 中文文件名** | `secure_filename(f.filename)` 后检查非空 | `secure_filename('测试视频.mp4')` → `'mp4'`（中文被清除），`Path('mp4').suffix` → `''`，中文用户上传被拒且错误信息不明确 | 新增回退逻辑：提取原始扩展名 + UUID 重命名 |

---

## Phase 1: 安全漏洞修复（Critical）

> 开源前 **必须完成**，否则存在远程代码执行、数据泄露、XSS 攻击风险。

### 1.1 文件上传路径穿越

- **风险等级**: 🔴 高危
- **位置**: `src/web/server.py` — `/api/upload_video` 路由
- **问题**: `f.filename` 未经净化直接拼接路径，攻击者可构造 `../../etc/cron.d/evil` 等文件名写入任意目录
- **修复方案**:
  ```python
  from werkzeug.utils import secure_filename
  import uuid

  @app.route("/api/upload_video", methods=["POST"])
  def api_upload_video():
      f = request.files.get("file")
      if f is None or f.filename == "":
          return jsonify({"ok": False, "error": "No file"}), 400

      # 1. 净化文件名
      # 注意: secure_filename 会清除非 ASCII 字符（如中文），
      # '测试视频.mp4' → 'mp4'，导致后续扩展名检查失败。
      # 对中文文件名需保留原始扩展名后用 UUID 重命名。
      raw_ext = Path(f.filename).suffix.lower() if f.filename else ""
      safe_name = secure_filename(f.filename)
      if not safe_name or not Path(safe_name).suffix:
          # secure_filename 清除了文件名主体（如纯中文名），用 UUID + 原始扩展名
          if raw_ext:
              safe_name = f"upload{raw_ext}"
          else:
              return jsonify({"ok": False, "error": "Invalid filename"}), 400

      # 2. 白名单扩展名
      ALLOWED_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
      ext = Path(safe_name).suffix.lower()
      if ext not in ALLOWED_EXT:
          return jsonify({"ok": False, "error": f"Unsupported format: {ext}"}), 400

      # 3. 限制文件大小 (Flask 层)
      # 在 app 配置中: app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

      # 4. UUID 重命名防覆盖
      unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
      upload_dir = Path("uploads")
      upload_dir.mkdir(exist_ok=True)
      save_path = (upload_dir / unique_name).resolve()

      # 5. 验证路径未逃逸
      if not save_path.is_relative_to(upload_dir.resolve()):
          return jsonify({"ok": False, "error": "Invalid path"}), 400

      f.save(str(save_path))
      # 仅返回相对路径，避免泄露服务器文件系统结构
      return jsonify({"ok": True, "path": str(Path("uploads") / unique_name)})
  ```
- **验证**: 用 `../../../tmp/test.mp4` 作为文件名测试，确认被拒绝

### 1.2 视频路径任意文件读取

- **风险等级**: 🔴 高危
- **位置**: `src/web/server.py` — `/api/start` 路由中 `path` 参数（约第1437行）
- **问题**: 客户端传入任意 `path` 直接传给 `VideoSource(path=path)` → `cv2.VideoCapture(path)`，可指向服务器任意可读文件
- **验证确认**: `VideoSource.__init__` 和 `open()` 内部无任何路径校验，仅检查 `isOpened()`
- **修复方案**:
  ```python
  # 在 api_start() 中视频模式分支:
  path = data.get("path", "")
  if not path:
      return jsonify({"ok": False, "error": "path is required"}), 400

  # 可配置的允许目录列表（默认仅 uploads/，可通过环境变量扩展）
  _allowed_video_dirs_str = os.environ.get("ALLOWED_VIDEO_DIRS", "uploads")
  ALLOWED_VIDEO_DIRS = [Path(d.strip()).resolve() for d in _allowed_video_dirs_str.split(",")]

  video_path = Path(path).resolve()
  if not any(video_path.is_relative_to(d) for d in ALLOWED_VIDEO_DIRS):
      return jsonify({"ok": False, "error": "视频路径不在允许的目录中"}), 403

  if not video_path.exists():
      return jsonify({"ok": False, "error": "文件不存在"}), 404
  ```
  > **设计说明**: 原方案仅硬编码 `uploads/` 目录，经验证会阻断合法的本地视频调试场景。
  > 改为可通过环境变量 `ALLOWED_VIDEO_DIRS` 配置允许的目录列表（逗号分隔），
  > 默认仅 `uploads/`，开发环境可设为 `uploads,/Users/xxx/videos`。
- **验证**: 用 `/etc/passwd`、`../secret.mp4` 等路径测试，确认被拒绝；设置环境变量后可访问配置的目录

### 1.3 模型路径注入

- **风险等级**: 🟠 中高危
- **位置**: `src/web/server.py` — `/api/start` 中 `model`（约第1208行）、`arcface_model`（约第1232行）参数
- **问题**: 客户端可指定任意模型路径，可能加载恶意 ONNX 文件或探测文件系统
- **验证确认**: 这两个参数直接从 `data.get()` 取值后传给 `SCRFDDetector(model_path=...)` 和 `ArcFaceEmbedder(model_path=...)`，无任何校验
- **修复方案**:
  ```python
  # 模型必须在 models/ 目录下，且必须是 .onnx 扩展名
  MODELS_DIR = Path("models").resolve()

  def _validate_model_path(path_str: str, allowed_ext: str = ".onnx") -> Path:
      """校验模型路径在 models/ 目录内且扩展名合法"""
      p = Path(path_str).resolve()
      if not p.is_relative_to(MODELS_DIR):
          raise ValueError("模型路径不在允许的目录中")  # 不暴露 MODELS_DIR 绝对路径
      if p.suffix.lower() != allowed_ext:
          raise ValueError(f"模型文件必须是 {allowed_ext} 格式")
      if not p.exists():
          raise FileNotFoundError("指定的模型文件不存在")  # 不回显用户输入的路径
      return p

  model = _validate_model_path(data.get("model", "models/det_10g.onnx"))
  arcface_model = _validate_model_path(data.get("arcface_model", "models/w600k_r50.onnx"))
  ```
  > **设计说明**: 原方案用文件名白名单过于死板，不支持用户添加新模型。
  > 改为目录约束 + 扩展名白名单，在安全与灵活性间取得平衡。

### 1.4 存储型 XSS — 前端 innerHTML 注入

- **风险等级**: 🔴 高危
- **位置**:
  - `src/web/templates/index_v2.html` — `renderOverlay()` 和 `renderCards()` 中 `name` 拼入 `innerHTML`
  - `src/web/templates/index.html` — `loadVideoList()` 中文件名拼入 `innerHTML` 和 `onclick`
- **问题**: `/api/person/rename` 写入的恶意 `name`（如 `<img onerror=alert(1)>`）会被直接渲染执行
- **修复方案（index_v2.html）**:
  ```javascript
  // 添加转义函数
  function escapeHtml(str) {
      const div = document.createElement('div');
      div.textContent = str;
      return div.innerHTML;
  }

  // renderOverlay 中所有用户可控字段必须转义:
  // 原: +'<div class="face-label-inner">'+name+emoji+'</div>'
  // 改: +'<div class="face-label-inner">'+escapeHtml(name)+emoji+'</div>'

  // renderCards 中同理:
  // 原: h+='<div class="person-name">'+displayName+'</div>';
  // 改: h+='<div class="person-name">'+escapeHtml(displayName)+'</div>';
  ```
- **修复方案（index.html）**:
  ```javascript
  // loadVideoList: 必须使用 DOM API 替代 innerHTML 拼接
  // 重要：仅用 escapeHtml 不够！onclick="selectVideo('"+f+"')" 中的 f
  // 同时存在 HTML 上下文和 JS 字符串上下文，需要双重转义或完全重构为事件绑定
  box.innerHTML = '';
  files.forEach(f => {
      const div = document.createElement('div');
      div.className = 'video-item';
      div.textContent = f;  // textContent 自动防 HTML 注入
      div.addEventListener('click', () => selectVideo(f));  // 事件绑定防 JS 注入
      box.appendChild(div);
  });
  ```
  > **设计说明**: 原方案仅建议 escapeHtml()，经验证不够——
  > `onclick="selectVideo('"+f+"')"` 同时涉及 HTML 属性上下文和 JS 字符串上下文，
  > 文件名如 `x');alert(1);//.mp4` 可绕过纯 HTML 转义。
  > 必须改用 `addEventListener` + `textContent` 彻底消除两种注入面。

- **修复方案（index_v2.html 的日志区域）**: `renderCards()` 中 `avatar` base64 拼入 `src="data:image/jpeg;base64,..."` 时，若 base64 被篡改可能逃逸属性引号。建议对 avatar 做正则白名单校验:
  ```javascript
  const safeAvatar = /^[A-Za-z0-9+/=]+$/.test(p.avatar) ? p.avatar : '';
  ```

- **后端辅助**: `/api/person/rename` 添加长度限制:
  ```python
  name = data.get("name", "").strip()[:50]  # 最大 50 字符
  ```
- **验证清单**:
  - 用 `<script>alert(1)</script>` 作为 name 测试 index_v2 → 确认被转义
  - 用 `"><img onerror=alert(1)>` 作为 name 测试 → 确认被转义
  - 用 `x');alert(1);//.mp4` 作为上传文件名测试 index.html → 确认 onclick 不触发
  - 用 `</div><img src=x onerror=alert(1)>.mp4` 作为文件名测试 → 确认 HTML 不注入

### 1.5 pickle 反序列化风险

- **风险等级**: 🟠 中危（已验证：模型路径为硬编码默认值，无法通过 API 远程指定）
- **位置**: `src/speaking/speaking_analyzer.py:79` — `pickle.load(f)`
- **问题**: `pickle.load` 可执行嵌入的任意 Python 代码
- **验证确认**:
  - `SpeakingAnalyzer.__init__` 的 `model_path` 参数默认为 `"models/speaking/speaking_model.pkl"`
  - `server.py:1456` 中调用 `SpeakingAnalyzer()` **未传入任何参数**，使用默认路径
  - `/api/start` 的 JSON 请求中 **无** `speaking_model` 或类似字段可控制此路径
  - **结论**: 攻击者无法通过网络 API 指定恶意 pkl 路径，风险限于本地文件被替换
- **风险重评**: 从原 🔴Critical 降为 🟠Medium——等同于「本地二进制替换」威胁模型
- **修复方案（彻底消除 pickle）**: 将模型拆分为安全格式，完全避免 pickle
  ```python
  # === 训练端 (train_speaking_model.py) ===
  # 1. XGBoost 模型用原生 JSON 格式保存（无 pickle）
  best_model.save_model("models/speaking/speaking_model.json")

  # 2. 元数据用 JSON 保存
  import json
  meta = {
      "model_type": best_name,
      "feature_cols": best["feature_cols"],
      "raw_features": RAW_FEATURES,
      "window_size": window,
  }
  with open("models/speaking/speaking_meta.json", "w") as f:
      json.dump(meta, f)

  # === 推理端 (speaking_analyzer.py) ===
  import xgboost as xgb
  import json

  # 加载 XGBoost 模型（原生格式，无反序列化风险）
  booster = xgb.Booster()
  booster.load_model("models/speaking/speaking_model.json")
  self._xgb_model = booster

  # 加载元数据（纯 JSON，无执行风险）
  with open("models/speaking/speaking_meta.json") as f:
      meta = json.load(f)
  self._feature_cols = meta["feature_cols"]
  self._raw_features = meta["raw_features"]
  self._window = meta["window_size"]
  ```
  > **设计说明**: 原方案建议改用 `joblib`，经验证 `joblib.load` 内部仍使用 pickle，
  > **不能**从根本上消除反序列化风险。SHA256 校验仅保证完整性，不阻止恶意 pickle。
  > 正确做法是**完全消除 pickle**：XGBoost 原生 JSON 格式 + 纯 JSON 元数据。
- **二次验证补充（模型类型与推理兼容性）**:
  - `train_speaking_model.py` 中 `best["best_model"]` 类型为 `xgb.XGBClassifier`（sklearn API）
  - `speaking_analyzer.py:204` 调用 `predict_proba(feat)[0, 1]` 取正类概率
  - 当最佳模型为 XGBoost 时 `scaler` 恒为 `None`，不影响 JSON 方案
  - 改用 `Booster.predict(DMatrix(feat))` 对二分类 logistic 目标返回正类概率标量，与原 `predict_proba[:,1]` 数值一致
  - **注意**: 若未来 LogisticRegression 胜出，现有代码中 `SpeakingAnalyzer` 未使用 scaler 做 transform，推理本身就不正确（既有缺陷，非本方案引入）
  ```python
  # === 推理端完整修改 (speaking_analyzer.py) ===
  import xgboost as xgb
  import json

  booster = xgb.Booster()
  booster.load_model("models/speaking/speaking_model.json")  # 原生格式，无 pickle
  self._xgb_booster = booster

  with open("models/speaking/speaking_meta.json") as f:
      meta = json.load(f)
  self._feature_cols = meta["feature_cols"]
  self._raw_features = meta["raw_features"]
  self._window = meta["window_size"]

  # 推理时（替换原 predict_proba）:
  feat_dm = xgb.DMatrix(feat, feature_names=self._feature_cols)
  prob = float(self._xgb_booster.predict(feat_dm)[0])  # binary:logistic → 正类概率
  ```
- **同步修改**: `train_speaking_model.py` 中:
  ```python
  # 替换 pickle.dump:
  best_model.get_booster().save_model(str(MODEL_OUT / "speaking_model.json"))
  meta = {"model_type": best_name, "feature_cols": best["feature_cols"],
          "raw_features": RAW_FEATURES, "window_size": window, "metrics": {...}}
  with open(MODEL_OUT / "speaking_meta.json", "w") as f:
      json.dump(meta, f)
  ```
- **验证**: 用同一特征向量对比旧 `XGBClassifier.predict_proba` 与新 `Booster.predict` 输出差值 ≤ 浮点误差

### 1.6 认证与访问控制

- **风险等级**: 🔴 高危（公网/局域网暴露时）
- **位置**: `src/web/server.py` — 全部路由
- **问题**: 所有 API 无任何认证，任何人可启停流水线、上传文件、修改身份信息
- **修复方案（最小化方案 — API Key）**:
  ```python
  import secrets
  import functools

  # 启动时生成或从环境变量读取
  API_KEY = os.environ.get("FACE_API_KEY", secrets.token_urlsafe(32))

  def require_api_key(f):
      @functools.wraps(f)
      def decorated(*args, **kwargs):
          key = request.headers.get("X-API-Key") or request.args.get("api_key")
          if key != API_KEY:
              return jsonify({"ok": False, "error": "Unauthorized"}), 401
          return f(*args, **kwargs)
      return decorated

  # 应用到管理类路由:
  @app.route("/api/start", methods=["POST"])
  @require_api_key
  def api_start(): ...

  @app.route("/api/stop", methods=["POST"])
  @require_api_key
  def api_stop(): ...

  @app.route("/api/upload_video", methods=["POST"])
  @require_api_key
  def api_upload_video(): ...

  @app.route("/api/person/rename", methods=["POST"])
  @require_api_key
  def api_person_rename(): ...
  ```
  - `/api/stats`、`/api/persons` 可按需加保护
  - 启动时打印 API Key 到控制台
  - 前端 JS 统一封装 `apiFetch()` 函数携带 Key（共 5 处 fetch 调用需修改）
  ```javascript
  // index_v2.html 中添加统一封装:
  const API_KEY = {{ api_key | tojson }};  // tojson 过滤器自动加引号并转义特殊字符
  function apiFetch(url, opts = {}) {
      opts.headers = Object.assign({'X-API-Key': API_KEY}, opts.headers || {});
      return fetch(url, opts);
  }
  // 所有 fetch('/api/...') 改为 apiFetch('/api/...')
  ```
- **二次验证补充（/video_feed 认证问题）**:
  `/video_feed` 通过 `<img src="/video_feed">` 加载，`<img>` 标签 **无法携带自定义 HTTP Header**，
  因此 API Key 方案 **无法保护 MJPEG 流**。解决方案（任选其一）:
  - **方案A（推荐 — 签名 URL）**: 后端生成带时效签名的 URL，前端设置 img src:
    ```python
    import hmac, time
    def signed_video_url():
        ts = str(int(time.time()))
        sig = hmac.new(API_KEY.encode(), ts.encode(), 'sha256').hexdigest()[:16]
        return f"/video_feed?ts={ts}&sig={sig}"

    # /video_feed 路由中验证:
    ts = request.args.get("ts", "")
    sig = request.args.get("sig", "")
    try:
        ts_int = int(ts)
    except (ValueError, TypeError):
        return "Bad Request", 400
    if abs(time.time() - ts_int) > 300:  # 5分钟有效期
        return "Expired", 403
    expected = hmac.new(API_KEY.encode(), ts.encode(), 'sha256').hexdigest()[:16]
    if not hmac.compare_digest(sig, expected):
        return "Forbidden", 403
    ```
  - **方案B（简单 — 仅绑定 127.0.0.1）**: 默认仅本地访问，`/video_feed` 不额外认证，在 README 中注明安全边界
  - **方案C（部署层）**: 通过 nginx 反向代理 + IP 白名单保护
- **验证**: 不带签名访问 `/video_feed` 确认返回 403；过期签名确认被拒绝

---

## Phase 2: 运行时缺陷修复（High）

> 影响程序稳定性的 Bug，应在发布前修复。

### 2.1 空帧崩溃（NoneType AttributeError）

- **位置**: `src/web/server.py` — `_generate_frames()` 约第1022行
- **问题**: `state._last_frame = frame.image` 在 `if frame is None` 检查 **之前** 执行
- **修复**:
  ```python
  # 原（错误顺序）:
  state._last_frame = frame.image    # frame 可能是 None
  if frame is None:                   # 来不及检查
      ...

  # 改为:
  if frame is None:
      if state.mode == "video":
          state.running = False
          break
      continue
  state._last_frame = frame.image
  ```

### 2.2 线程竞态 — 共享字典无锁

- **位置**: `src/web/server.py` — `state.track_to_person`、`state.track_similarities` 等
- **问题**: 主线程（`_generate_frames`）读取、`IdentityWorker` 后台线程（`_alignment_step`）写入，无锁保护
- **验证确认**:
  - `PipelineState.__init__` 中定义了 `self.lock = threading.Lock()`（约第204行）
  - **但 `state.lock` 在整个 `server.py` 中从未被使用**（零次 `with state.lock`）
  - `_alignment_step` 中写入 `state.track_to_person` 约 7 处（第696、705、720、730、751、762、830行）
  - `_generate_frames` 中读取 `state.track_to_person` 约 2 处（第1078、1113行）
- **风险重评**: CPython GIL 保护单次 `dict.__setitem__`/`__getitem__` 不会段错误，
  主要风险是 **逻辑一致性**（某帧读到部分更新的映射），**非内存安全问题**
- **修复方案（原子快照替换，低开销）**:
  ```python
  # 方案: IdentityWorker 构建完整快照后原子替换引用，主线程读引用即可

  # _alignment_step 末尾，所有 track 处理完成后:
  # 构建本帧的完整映射快照
  new_tp = dict(state.track_to_person)    # 浅拷贝当前状态
  new_ts = dict(state.track_similarities)
  # 原子替换（Python 赋值操作在 GIL 下是原子的）
  state._track_to_person_snapshot = new_tp
  state._track_similarities_snapshot = new_ts

  # _generate_frames 中读取时使用快照:
  tp = state._track_to_person_snapshot    # 读引用，无锁
  ts = state._track_similarities_snapshot
  # 后续用 tp, ts
  ```
  > **设计说明**: 原方案对每次读写加 `state.lock`，在 30fps 视频流 + 每帧多次字典操作下
  > 可能造成性能瓶颈。改为「后台线程构建完整快照后原子替换引用」模式，
  > 主线程只读引用（单次赋值在 GIL 下原子），开销趋近于零，且保证逻辑一致性。
- **二次验证补充（快照需覆盖的完整字段列表）**:
  经验证，`_alignment_step` 中 **写入** 且 `_generate_frames` 中 **读取** 的共享字典不止两个:
  | 字典 | worker写入 | 主线程读取位置 |
  |------|-----------|---------------|
  | `track_to_person` | 7处（696,705,720,730,751,762,830行） | _persons_cache构建(1078) + draw_tracks(1113) |
  | `track_similarities` | 同上对应位置 | draw_tracks(1113) |
  | `person_identity_states` | M5分支更新 | _persons_cache构建(1080) + draw_tracks(1119) |
  | `person_to_registered` | M5/合并逻辑 | _persons_cache构建(1081) + draw_tracks(1121) |

  快照应覆盖以上 **全部4个字典**。

  > **三次验证修正**: `_identity_snapshot` 必须在 `PipelineState.__init__` 中初始化为空快照，
  > 否则 `_generate_frames` 在首帧（`_alignment_step` 尚未执行时）读取会触发 `AttributeError`。
  ```python
  # PipelineState.__init__ 中添加:
  self._identity_snapshot = {
      "track_to_person": {},
      "track_similarities": {},
      "person_identity_states": {},
      "person_to_registered": {},
  }
  ```

  ```python
  # _alignment_step 末尾:
  state._identity_snapshot = {
      "track_to_person": dict(state.track_to_person),
      "track_similarities": dict(state.track_similarities),
      "person_identity_states": dict(state.person_identity_states),
      "person_to_registered": dict(state.person_to_registered),
  }

  # _generate_frames 中读取:
  snap = state._identity_snapshot  # 读一个引用（原子）
  tp = snap["track_to_person"]
  ts = snap["track_similarities"]
  # ... 传给 draw_tracks 和 _persons_cache 构建
  ```
  标量统计（`align_ms`、`embed_extracted` 等）在 GIL 下单次赋值安全，无需纳入快照。
- **额外风险**: `_stop_pipeline()` 调用 `state.reset()` 清空字典时，旧的 `_generate_frames` 循环可能仍在读取。`reset()` 应先设 `state.running = False` 并等待流结束（已部分实现），或在快照模式下 `reset` 只需替换快照引用为空字典即可。

### 2.3 STrack ID 生成 ~~非线程安全~~ （已验证：当前无风险）

- **位置**: `src/tracking/track.py` — `STrack._next_id` 类变量，`_alloc_id()` 方法
- **验证确认**:
  - `_alloc_id()` 仅在 `activate()` 中调用（分配新轨迹 ID）
  - `activate()` 仅在 `BoTSORTTracker.update()` 中调用
  - `update()` 仅在 `tracker.step()` 中调用
  - `tracker.step()` 仅在 `_generate_frames()` 主循环中调用（约第1039行）
  - `IdentityWorker` 和 `MouthWorker` **不调用** `tracker.step()` 或 `STrack` 任何方法
  - **结论: 在当前架构下，ID 生成始终在单一主线程中执行，不存在并发竞争**
- **风险重评**: 从原 🟠High 降为 🔵Info — 当前无实际风险
- **建议（防御性编程，非必须）**: 若未来可能引入并行 tracker，可预防性改用 `itertools.count()`:
  ```python
  import itertools
  class STrack:
      _id_counter = itertools.count(1)

      @classmethod
      def _alloc_id(cls):
          return next(cls._id_counter)
  ```
  > **优先级**: 低。当前单线程调用，不改也不影响正确性和安全性。

### 2.4 查询参数类型转换未捕获

- **位置**: `src/web/server.py` — `/api/log`、`/api/embed_log`、`/api/person/rename`
- **问题**: `int(request.args.get("n", 50))` 和 `int(pid)` 无 try/except，非法输入直接 500 崩溃
- **验证确认**: 这些路由不在 `/api/start` 的大 try/except 中，异常完全未捕获
- **修复**:
  ```python
  # /api/log, /api/embed_log:
  try:
      n = int(request.args.get("n", 50))
      n = max(1, min(500, n))
  except (ValueError, TypeError):
      n = 50

  # /api/person/rename:
  try:
      pid = int(data.get("person_id"))
  except (ValueError, TypeError):
      return jsonify({"ok": False, "error": "person_id must be integer"}), 400
  ```

### 2.5 API 响应泄露文件系统路径

- **位置**: `src/web/server.py` — `/api/stats`、`/api/start`、`/api/log` 响应中的 `source_id` 字段
- **问题**: `VideoSource.source_id` 格式为 `file:/Users/xxx/video.mp4`，包含完整的本地绝对路径，在正常 200 响应中返回给客户端
- **验证确认**: `src/ingestion/video_source.py` 中 `self.source_id = f"file:{path}"`，path 为完整传入路径
- **修复**: 仅返回文件名而非完整路径:
  ```python
  # video_source.py:
  self.source_id = f"file:{Path(path).name}"  # 只保留文件名
  ```

### 2.6 RegisteredPersonDB.save() 非原子写入

- **位置**: `src/embedding/identity_state.py` — `save()` 方法
- **问题**: 先写 `templates.npz` 再写 `metadata.json`，中间崩溃会导致数据不一致或文件损坏
- **验证确认**: 无临时文件 + rename 的原子写入模式
- **修复**:
  ```python
  def save(self):
      os.makedirs(self._db_dir, exist_ok=True)
      templates_path = os.path.join(self._db_dir, "templates.npz")
      meta_path = os.path.join(self._db_dir, "metadata.json")

      # 写入临时文件，成功后原子替换
      tmp_tpl = templates_path + ".tmp"
      tmp_meta = meta_path + ".tmp"
      np.savez(tmp_tpl, ...)
      with open(tmp_meta, "w", encoding="utf-8") as f:
          json.dump(meta_out, f, ensure_ascii=False, indent=2)

      os.replace(tmp_tpl, templates_path)   # 原子替换
      os.replace(tmp_meta, meta_path)        # 原子替换
  ```

### 2.7 错误信息泄露

- **位置**: `src/web/server.py` — 多处 `return jsonify({"error": str(e)})`
- **修复**: 生产环境返回泛化错误，详细信息仅写日志:
  ```python
  import logging
  logger = logging.getLogger(__name__)

  except Exception as e:
      logger.exception("ArcFace model load failed")
      return jsonify({"ok": False, "error": "模型加载失败，请检查服务端日志"}), 500
  ```

### 2.8 安全响应头

- **位置**: `src/web/server.py`
- **修复**: 添加全局安全响应头:
  ```python
  @app.after_request
  def add_security_headers(response):
      response.headers["X-Content-Type-Options"] = "nosniff"
      response.headers["X-Frame-Options"] = "SAMEORIGIN"
      response.headers["X-XSS-Protection"] = "1; mode=block"
      response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
      # CSP: 根据实际引用的外部资源调整
      # CSP 说明: 当前前端使用内联 <style> 和 <script>，
      # 需要 'unsafe-inline'；长期目标是将 JS/CSS 外置后移除 unsafe-inline。
      # 即使有 unsafe-inline，CSP 仍能阻止 base-uri 劫持和 object 嵌入等攻击。
      response.headers["Content-Security-Policy"] = (
          "default-src 'self'; "
          "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
          "font-src 'self' https://fonts.gstatic.com; "
          "img-src 'self' data: blob:; "
          "script-src 'self' 'unsafe-inline'; "
          "object-src 'none'; "
          "base-uri 'self'; "
          "form-action 'self'"
      )
      return response
  ```

### 2.9 请求体大小限制

- **位置**: `src/web/server.py`
- **问题**: 文件上传和 JSON body 均无大小限制，可通过超大请求耗尽内存/磁盘
- **修复**: 添加全局配置:
  ```python
  app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB（覆盖上传和JSON body）
  ```

### 2.10 /video_feed 多连接 DoS

- **位置**: `src/web/server.py` — `/video_feed` 路由
- **问题**: 每个客户端连接都会持有一个 `_generate_frames()` 生成器，多线程 Flask 下多个连接并行执行推理和 JPEG 编码，无连接数限制，可耗尽 CPU 和带宽
- **修复**: 限制并发流连接数:
  ```python
  import threading
  _stream_lock = threading.Lock()
  _active_streams = 0
  _max_streams = 3

  @app.route("/video_feed")
  def video_feed():
      global _active_streams
      with _stream_lock:
          if _active_streams >= _max_streams:
              return "Too many streams", 429
          _active_streams += 1
      def gen():
          try:
              yield from _generate_frames()
          finally:
              global _active_streams
              with _stream_lock:
                  _active_streams -= 1
      return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
  ```
  > **三次验证修正**: 原方案 `_active_streams += 1` 非原子操作（LOAD + ADD + STORE 三条字节码），
  > 多线程 Flask 下计数器本身存在竞态。必须用 `threading.Lock` 保护。

---

## Phase 3: 开源基础设施

> 开源项目的必备文件和配置。

### 3.1 添加 LICENSE 文件

- **推荐**: MIT License（宽松，利于传播）或 Apache-2.0（含专利授权）
- 创建 `LICENSE` 文件于项目根目录

### 3.2 创建 .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/

# 模型文件（大体积，不入库）
models/*.onnx
models/*.task
models/speaking/*.onnx
models/speaking/*.pkl

# 运行时生成
output/
uploads/
data/recordings/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
__MACOSX/
Thumbs.db
```

### 3.3 完善 requirements.txt

拆分为两个文件:

**requirements.txt（核心运行）**:
```
opencv-python>=4.8.0
numpy>=1.24.0
onnxruntime>=1.16.0
flask>=3.0.0
scipy>=1.11.0
mediapipe>=0.10.0
xgboost>=2.0.0
scikit-learn>=1.3.0
Pillow>=10.0.0
```

**requirements-training.txt（训练与录制工具）**:
```
-r requirements.txt
pandas>=2.0.0
keyboard>=0.13.0
```

### 3.4 更新 README.md

需要修改的内容:
- [ ] 替换 `.\.venv312\Scripts\Activate.ps1` 为跨平台说明（Unix + Windows）
- [ ] 添加 LICENSE 声明段落
- [ ] 添加"安全说明"章节（说明默认仅绑定 127.0.0.1）
- [ ] 添加"贡献指南"链接
- [ ] 添加模型下载说明（模型文件不入库，需要用户自行下载）
- [ ] 添加"隐私与合规"声明（涉及人脸生物特征数据）

### 3.5 添加 CONTRIBUTING.md

包含: 开发环境搭建、代码风格、提交规范、PR 流程。

### 3.6 添加 SECURITY.md

包含: 安全漏洞报告方式（建议私有渠道，非公开 Issue）。

---

## Phase 4: 代码质量提升

> 提高可维护性、可读性和健壮性。

### 4.1 统一日志系统

- **范围**: 全项目 `print(...)` → `logging.getLogger(__name__)`
- **影响文件**: `src/web/server.py`（约30处）、`src/detectors/scrfd_detector.py`、`src/embedding/identity_state.py`、`src/speaking/mouth_worker.py` 等
- **方案**: 在 `run_web_v2.py` 入口统一配置 `logging.basicConfig`，各模块用 `logger = logging.getLogger(__name__)`
- **优先级**: 中（不影响功能，但大幅提升可维护性）

### 4.2 输入参数校验强化

- **位置**: `src/web/server.py` — `/api/start`
- **内容**:
  ```python
  # 对所有数值参数添加边界校验
  det_thresh = max(0.01, min(1.0, float(data.get("det_thresh", 0.5))))
  det_size = max(320, min(1280, int(data.get("det_size", 640))))
  max_embed_per_frame = max(1, min(20, int(data.get("max_embed_per_frame", 5))))
  # ... 所有数值参数同理

  # 对非数值参数做类型检查
  if data.get("mode") not in ("camera", "video"):
      return jsonify({"ok": False, "error": "mode must be 'camera' or 'video'"}), 400
  ```

### 4.3 移除调试代码

- **位置**: `src/web/server.py` 第1199-1203行
- **内容**: 删除 `print("=" * 60)` 及完整请求体打印，或改为 `logger.debug`:
  ```python
  logger.debug("[/api/start] 请求参数: %s", json.dumps(data, ensure_ascii=False))
  ```

### 4.4 补全类型注解

- **高优先级文件**:
  - `src/web/server.py`: `PipelineState` 成员变量、`_alignment_step` 参数
  - `src/tracking/track.py`: `xyxy_to_cxywh`、`cxywh_to_xyxy`
  - `src/tracking/kalman_filter.py`: `initiate`、`predict`、`update` 返回值
  - `src/embedding/person_registry.py`: 方法参数和返回值

### 4.5 文档与实现对齐

| 文件 | 问题 | 修复 |
|------|------|------|
| `src/embedding/candidate_pool.py` | 头部文档说仅 UNKNOWN_STRONG 进入，但代码允许 AMBIGUOUS | 更新文档说明两种情况 |
| `src/speaking/mouth_analyzer.py` | 文档提到 pitch 检测但未实现 | 删除 pitch 相关文档或补充实现 |
| `src/speaking/speaking_analyzer.py` | 文档说 yaw/pitch/roll 自遮挡，实际仅 yaw > 60 | 对齐文档 |

### 4.6 清除死代码和未使用导入

| 文件 | 问题 |
|------|------|
| `src/tracking/mouth_tracker.py:18` | `import math` 未使用 |
| `run_ingestion.py:26` | `import io` 未使用 |
| `src/speaking/mouth_analyzer.py` | 已被 `speaking_analyzer.py` 替代，README 也标注"已被替代" |

### 4.7 消除同名类混淆

- `MouthState` 在 `tracking/mouth_tracker.py` 和 `speaking/mouth_analyzer.py` 中同名
- **方案**: 将 `tracking/mouth_tracker.py` 中的重命名为 `TrackMouthState`

---

## Phase 5: 架构重构

> 提升项目长期可维护性，可分批执行。

### 5.1 拆分 server.py（1804行 → 多文件）

```
src/web/
├── server.py          # Flask app 创建、路由注册（~200行）
├── routes/
│   ├── api.py         # /api/* REST 路由
│   ├── stream.py      # /video_feed MJPEG 流
│   └── upload.py      # 文件上传
├── pipeline.py        # PipelineState + 主循环逻辑
├── identity_worker.py # IdentityWorker 类
├── config.py          # CreditGateConfig, Module5Config 等 dataclass
└── templates/
    ├── index_v2.html
    └── index.html
```

### 5.2 去全局状态

- `PipelineState` 从模块级单例改为 Flask app context 或显式注入
- 便于未来支持多实例、测试 mock

### 5.3 消除跨线程直接共享

- `IdentityWorker` 的结果通过 `queue.Queue` 返回，主线程在安全点合并
- 消除对 `state.track_to_person` 等字典的跨线程直写

### 5.4 硬编码路径配置化

- 所有默认路径（`models/`、`output/`、`data/`）提取到配置文件或环境变量
- 创建 `config.py` 或使用 `pydantic-settings`:
  ```python
  class AppConfig:
      MODELS_DIR: str = "models"
      OUTPUT_DIR: str = "output"
      UPLOAD_DIR: str = "uploads"
      DET_MODEL: str = "det_10g.onnx"
      ARCFACE_MODEL: str = "w600k_r50.onnx"
  ```

---

## Phase 6: 工程化与 CI/CD

> 可选但推荐，提升开源项目的专业度。

### 6.1 添加 pyproject.toml

统一项目元数据、构建配置、工具配置:
```toml
[project]
name = "face-recognition-system"
version = "1.0.0"
requires-python = ">=3.12"

[tool.ruff]
line-length = 120
select = ["E", "W", "F", "I"]

[tool.mypy]
python_version = "3.12"
warn_return_any = true
```

### 6.2 GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install ruff mypy
      - run: ruff check src/
      - run: mypy src/ --ignore-missing-imports
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/ -v
```

### 6.3 添加单元测试

优先覆盖:
- `src/alignment/quality.py` — 质量评估逻辑
- `src/embedding/candidate_pool.py` — 候选池状态机
- `src/tracking/matching.py` — 匹配算法
- Web 路由输入校验

### 6.4 跨平台兼容

| 问题 | 位置 | 修复 |
|------|------|------|
| Windows 字体硬编码 | `record_speaking_data.py:173` | 使用 `matplotlib.font_manager` 或打包字体文件 |
| `CAP_DSHOW` Windows Only | `camera_source.py:32` | ✅ 已修复为 `CAP_ANY` |
| PowerShell 激活脚本 | `README.md` | 改为跨平台说明 |

### 6.5 隐私合规声明

项目涉及人脸生物特征数据，开源时需要:
- [ ] README 中添加"隐私与数据"章节
- [ ] 说明数据仅本地处理，不上传到任何服务器
- [ ] 说明 `data/recordings/` 中的训练数据不包含在仓库中
- [ ] 建议用户遵守当地人脸识别相关法律法规（如 GDPR、中国《个人信息保护法》）

---

## 附录 A: 完整问题清单

| # | 严重程度 | Phase | 文件 | 行号 | 问题摘要 | 已交叉验证 |
|---|----------|-------|------|------|----------|------------|
| 1 | 🔴 Critical | 1.1 | server.py | 1792 | 路径穿越 — filename 未净化 | ✅ `save_path = upload_dir / f.filename` 无 secure_filename |
| 2 | 🔴 Critical | 1.2 | server.py | 1437 | 视频路径任意文件可控 | ✅ `VideoSource(path=path)` 无校验，path 完全来自客户端 JSON |
| 3 | 🟠 High | 1.3 | server.py | 1208,1232 | 模型路径来自客户端 | ✅ `data.get("model")` 直接传入 ONNX 加载 |
| 4 | 🔴 Critical | 1.4 | index_v2.html | 645,696 | innerHTML 存储型 XSS (name) | ✅ `p.name` 经 `/api/person/rename` 可写入，未转义拼入 innerHTML |
| 5 | 🔴 Critical | 1.4 | index.html | 636-637 | innerHTML + onclick 双上下文 XSS | ✅ 文件名拼入 HTML 和 JS 字符串，需 DOM API 重构而非仅 escapeHtml |
| 6 | 🟠 High | 1.5 | speaking_analyzer.py | 79 | pickle.load（本地路径不可远程控制） | ✅ 路径硬编码默认值，API 无法指定；风险为本地文件替换 |
| 7 | 🔴 Critical | 1.6 | server.py | 全部路由 | 无认证/授权 | ✅ 无 login/token/key，任意客户端可操作 |
| 8 | 🟠 High | 2.1 | server.py | 1022 | frame=None 时访问 .image | ✅ `state._last_frame = frame.image` 在 None 检查之前 |
| 9 | 🟡 Medium | 2.2 | server.py | 696,830,1078,1113 | 跨线程字典无锁（逻辑一致性） | ✅ state.lock 定义但从未使用；GIL 保护单操作不段错误 |
| 10 | 🔵 Info | 2.3 | track.py | 52-61 | _next_id 类变量递增 | ✅ 已验证仅主线程调用 tracker.step()，当前无并发风险 |
| 11 | 🟡 Medium | 2.4 | server.py | 1699,1778,1617 | int() 未捕获 → 500 崩溃 | ✅ 第二轮新发现 |
| 12 | 🟡 Medium | 2.5 | video_source.py | source_id | API 响应泄露文件系统绝对路径 | ✅ 第二轮新发现 |
| 13 | 🟡 Medium | 2.6 | identity_state.py | save() | 非原子写入，崩溃可损坏数据 | ✅ 第二轮新发现 |
| 14 | 🟠 High | 2.7 | server.py | 多处 | str(e) 返回客户端泄露信息 | ✅ |
| 15 | 🟡 Medium | 2.8 | server.py | — | 缺安全响应头 | ✅ |
| 16 | 🟠 High | 2.9 | server.py | — | 无请求体大小限制 | ✅ |
| 17 | 🟡 Medium | 2.10 | server.py | /video_feed | 无并发流限制 → CPU DoS | ✅ 第二轮新发现 |
| 18 | 🟡 Medium | 3.3 | requirements.txt | — | 依赖声明不完整（仅5包，实需11+） | ✅ |
| 19 | 🟡 Medium | 3.1 | — | — | 缺 LICENSE 文件 | ✅ |
| 20 | 🟡 Medium | 3.2 | — | — | 缺 .gitignore | ✅ |
| 21 | 🟡 Medium | 3.4 | README.md | 8-9 | Windows 特定路径 (.venv312) | ✅ |
| 22 | 🟡 Medium | 4.1 | 全项目 | ~30处 | print 替代 logging | ✅ |
| 23 | 🟡 Medium | 4.2 | server.py | /api/start | 参数无边界校验 | ✅ |
| 24 | 🟡 Medium | 4.3 | server.py | 1199-1203 | 调试打印未移除 ("便于验收") | ✅ |
| 25 | 🔵 Low | 4.4 | 多文件 | — | 类型注解不完整 | ✅ |
| 26 | 🔵 Low | 4.5 | 3文件 | — | 文档与实现不一致 | ✅ |
| 27 | 🔵 Low | 4.6 | 2文件 | — | 未使用的 import | ✅ |
| 28 | 🔵 Low | 4.7 | 2文件 | — | MouthState 同名混淆 | ✅ |
| 29 | 🔵 Low | 5.1 | server.py | — | 1804行单文件过大 | ✅ |
| 30 | 🔵 Low | 5.4 | 多文件 | — | 硬编码默认路径 | ✅ |

---

## 附录 B: 文件影响矩阵

| 文件 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|------|---------|---------|---------|---------|---------|
| `src/web/server.py` | ✅ 1.1,1.2,1.3,1.6 | ✅ 2.1-2.6 | | ✅ 4.1-4.3 | ✅ 5.1-5.4 |
| `src/web/templates/index_v2.html` | ✅ 1.4 | | | | |
| `src/web/templates/index.html` | ✅ 1.4 | | | | |
| `src/speaking/speaking_analyzer.py` | ✅ 1.5 | | | ✅ 4.5 | |
| `train_speaking_model.py` | ✅ 1.5 | | | | |
| `src/tracking/track.py` | | ✅ 2.3 | | ✅ 4.4 | |
| `requirements.txt` | | | ✅ 3.3 | | |
| `README.md` | | | ✅ 3.4 | | |
| `src/tracking/mouth_tracker.py` | | | | ✅ 4.6,4.7 | |
| `run_ingestion.py` | | | | ✅ 4.6 | |
| `record_speaking_data.py` | | | | | ✅ 6.4 |

---

## 执行建议

1. **Phase 1 + Phase 2**: 开源前必须完成，预计工作量 2-3 天
2. **Phase 3**: 与 Phase 1 并行，预计 1 天
3. **Phase 4**: 开源后可持续改进，预计 2-3 天
4. **Phase 5**: 中长期重构，可分 PR 逐步推进
5. **Phase 6**: 可选，按需推进

> 建议在 Phase 1-3 完成后即可进行首次开源发布（alpha/beta），后续 Phase 作为迭代改进。
