# 测试失败原因分析

## 测试用例
`BackgroundPythonEventLoop schedules coroutines without blocking`

## 失败现象

1. **断言失败**：在 `test_python_runtime.cpp:34`，`status == std::future_status::ready` 失败
   - 期望：`std::future_status::ready`
   - 实际：`std::future_status::timeout`（等待 1 秒后仍未完成）

2. **程序崩溃**：`terminate called without an active exception`
   - 发生在 fallback 线程执行协程时

## 根本原因分析

### 问题 1：测试未初始化事件循环

**位置**：`test/execution/distributed/test_python_runtime.cpp:24`

```cpp
auto &loop = BackgroundPythonEventLoop::Get();
auto fut = loop.Schedule(coro);  // 直接调用 Schedule，没有先调用 Start()
```

**问题**：
- 测试直接调用 `Schedule()`，但没有先调用 `Start()` 来初始化后台事件循环
- `loop_` 成员变量在构造函数中是未初始化的（默认构造的 `py::object()`）
- 导致 `Schedule()` 无法使用主事件循环，只能回退到 fallback 路径

### 问题 2：Schedule() 方法中的线程安全问题

**位置**：`src/execution/distributed/python/python_runtime.cpp:264`

```cpp
py::gil_scoped_acquire gil;
if (!loop_.is_none()) {  // ⚠️ 访问 loop_ 时没有持有 mutex_
    // ...
}
```

**问题**：
- `loop_` 在头文件中标注为 `protected by mutex_`，但 `Schedule()` 方法访问它时没有持有锁
- 存在数据竞争：`Start()` 线程可能在设置 `loop_` 的同时，`Schedule()` 在读取它
- 从输出看：`loop_.is_none=false` 但 `loop_.ptr() == nullptr`，说明读取到了不一致的状态

### 问题 3：Fallback 路径中的崩溃

**位置**：`src/execution/distributed/python/python_runtime.cpp:59`

**崩溃点**：
```
execute_python_coroutine: about to call asyncio.new_event_loop (ts=1194290872)
terminate called without an active exception
```

**可能原因**：
1. **Python 版本兼容性问题**：代码使用 Python 3.6（从输出 `/usr/lib64/python3.6/asyncio/__init__.py` 可见）
   - Python 3.6 中 `asyncio.run()` 不存在，代码走 fallback 路径
   - 在 fallback 路径中创建新的事件循环时可能存在问题

2. **异常处理不完整**：
   - `execute_python_coroutine()` 中的异常可能没有被正确捕获
   - 析构函数中抛出异常会导致 `std::terminate` 被调用

3. **GIL 状态问题**：
   - Fallback 线程在获取 GIL 后执行，但可能在某个时刻 GIL 状态不一致

### 问题 4：Future 超时

**位置**：`test/execution/distributed/test_python_runtime.cpp:33`

```cpp
auto status = fut.wait_for(std::chrono::seconds(1));
REQUIRE(status == std::future_status::ready);
```

**问题**：
- 由于 fallback 线程在执行协程时崩溃，promise 从未被设置值
- Future 一直处于 pending 状态，1 秒后超时

## 修复建议

### 修复 1：在测试中调用 Start()

```cpp
auto &loop = BackgroundPythonEventLoop::Get();
loop.Start();  // 添加这一行
auto fut = loop.Schedule(coro);
```

### 修复 2：在 Schedule() 中正确使用锁

```cpp
std::future<std::optional<py_object_t>> BackgroundPythonEventLoop::Schedule(py_object_t coro) {
    auto prom = std::make_shared<std::promise<std::optional<py::object>>>();
    auto fut = prom->get_future();
    try {
        py::gil_scoped_acquire gil;
        py::object loop_copy;
        {
            std::lock_guard<std::mutex> guard(mutex_);  // 添加锁
            if (loop_.is_none() || loop_.ptr() == nullptr) {
                // 如果循环未初始化，尝试启动它
                // 或者直接进入 fallback 路径
            } else {
                loop_copy = loop_;  // 复制 loop_ 的引用
            }
        }
        // 使用 loop_copy 而不是直接使用 loop_
        // ...
    } catch (...) {
        // ...
    }
    // fallback path...
}
```

### 修复 3：改进异常处理

在 `execute_python_coroutine()` 中确保所有异常都被捕获，特别是在创建事件循环时。

### 修复 4：确保 Promise 总是被设置

即使在崩溃情况下，也应该设置 promise 以避免 future 永远挂起。

## 调试建议

1. 添加更多日志来追踪事件循环的初始化过程
2. 使用 gdb 或 valgrind 来定位崩溃的确切位置
3. 检查 Python 版本兼容性，考虑升级到 Python 3.7+（支持 `asyncio.run()`）
4. 验证 GIL 的获取和释放是否正确

