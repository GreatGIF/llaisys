-- ==========================================
-- 辅助函数：收集 MACA SDK 路径
-- ==========================================
local function _collect_maca_roots()
    local roots = {}

    local maca_sdk = get_config("maca-sdk")
    if maca_sdk and maca_sdk ~= "" then
        table.insert(roots, maca_sdk)
    end

    local env_vars = {
        "MACA_PATH",
        "MXMACA_HOME",
        "MACA_HOME",
        "MACA_SDK_HOME",
        "MX_HOME",
        "MXSDK_HOME"
    }
    for _, env_name in ipairs(env_vars) do
        local p = os.getenv(env_name)
        if p and p ~= "" then
            table.insert(roots, p)
        end
    end

    local default_roots = {
        "/opt/maca",
        "/usr/local/maca",
        "/opt/mx",
        "/usr/local/mx"
    }
    for _, p in ipairs(default_roots) do
        table.insert(roots, p)
    end

    local filtered = {}
    local exists = {}
    for _, p in ipairs(roots) do
        if not exists[p] and os.isdir(p) then
            table.insert(filtered, p)
            exists[p] = true
        end
    end
    return filtered
end

-- ==========================================
-- 辅助函数：定位 mxcc 可执行文件
-- ==========================================
local function _resolve_mxcc_path()
    -- 1. 先从 PATH 中查找
    local path_env = os.getenv("PATH") or ""
    for bindir in string.gmatch(path_env, "([^:]+)") do
        local candidate = path.join(bindir, "mxcc")
        if os.isfile(candidate) then
            return candidate
        end
    end

    -- 2. 从 MACA SDK 路径中查找
    for _, root in ipairs(_collect_maca_roots()) do
        local candidates = {
            path.join(root, "mxgpu_llvm/bin/mxcc"),
            path.join(root, "bin/mxcc")
        }
        for _, candidate in ipairs(candidates) do
            if os.isfile(candidate) then
                return candidate
            end
        end
    end
    return nil
end

-- ==========================================
-- 自定义 toolchain: mxcc
--
-- 核心设计：
--   mxcc 是 MetaX 的 GPU 编译器，能编译 .cu 文件（CUDA 兼容）。
--   但是 xmake 内置的 CUDA 语言流程会强制检测 NVIDIA CUDA SDK：
--     .cu → language("cuda") → rule("cuda") → rule("cuda.env")
--       → assert(find_cuda(...), "Cuda SDK not found!")
--
--   为了绕开这个问题，我们采用如下策略：
--   1. 将 .cu 文件标记为 sourcekind = "cxx"（C++ 源文件）
--      这样 xmake 不会加载 CUDA 语言模块，不会触发 CUDA SDK 检测
--   2. 将 mxcc 设置为 C++ 编译器（cxx toolset）
--      mxcc 本身就能理解 .cu 文件中的 CUDA 语法
--   3. 链接器也使用 mxcc，确保 GPU 代码正确链接
-- ==========================================
toolchain("mxcc")
    set_kind("standalone")
    set_description("MetaX MXCC toolchain for CUDA-compatible .cu build")

    on_check(function (toolchain)
        local mxcc = _resolve_mxcc_path()
        if not mxcc then
            cprint("${yellow}Warning: mxcc not found in PATH or MACA SDK paths.")
            cprint("${yellow}Please set MACA_PATH or add mxcc to PATH.")
            return false
        end
        return true
    end)

    on_load(function (toolchain)
        local mxcc = _resolve_mxcc_path()
        if not mxcc then
            mxcc = "mxcc" -- fallback, 让系统 PATH 去找
        end

        local mxcc_wrapper = path.join(os.projectdir(), "scripts/mxcc_wrapper.sh")
        if not os.isfile(mxcc_wrapper) then
            raise("mxcc wrapper not found: " .. mxcc_wrapper)
        end

        toolchain:add("runenvs", "MXCC_REAL", mxcc)

        -- 我们必须使用 gcc@ 或 clang@ 前缀，否则 xmake 不知道如何给编译器传参（-I, -D 等）
        -- 这里使用 gcc@ 前缀，但通过 mapflags 吃掉 mxcc 不支持的特有 flag
        toolchain:set("toolset", "cxx", "gcc@" .. mxcc_wrapper)
        toolchain:set("toolset", "cc",  "gcc@" .. mxcc_wrapper)
        toolchain:set("toolset", "ld",  "gcc@" .. mxcc_wrapper)
        toolchain:set("toolset", "sh",  "gcc@" .. mxcc_wrapper)
        toolchain:set("toolset", "ar",  "ar")

        -- mxcc 无法识别 gcc 特有的部分 flags，我们通过 mapflags 将其映射为空白
        toolchain:set("mapflags", {
            ["-finput-charset=UTF-8"] = "",
            ["-fexec-charset=UTF-8"]  = "",
            ["-Qunused-arguments"]    = "",
            ["-s"]                    = "",
            -- xmake may append "-x c++" for sourcekind=cxx; rewrite it to MACA mode
            ["-x c++"]                = "-x maca"
        })


        -- 添加 MACA SDK 的 include/lib 路径
        for _, root in ipairs(_collect_maca_roots()) do
            local include_candidates = {
                path.join(root, "include"),
                path.join(root, "include/maca"),
                path.join(root, "include/mx"),
                path.join(root, "include/mcblas"),
                path.join(root, "mxgpu_llvm/include"),
                path.join(root, "tools/cu-bridge/include"),
                path.join(root, "tools/cu-bridge/include/bridge/runtime"),
                path.join(root, "tools/cu-bridge/include/bridge/blas"),
                path.join(root, "include/common"),
            }
            for _, inc in ipairs(include_candidates) do
                if os.isdir(inc) then
                    toolchain:add("includedirs", inc)
                end
            end

            local lib_candidates = {
                path.join(root, "lib64"),
                path.join(root, "lib"),
                path.join(root, "lib/x64")
            }
            for _, libdir in ipairs(lib_candidates) do
                if os.isdir(libdir) then
                    toolchain:add("linkdirs", libdir)
                end
            end
        end
    end)
toolchain_end()

-- ==========================================
-- 可选的用户配置项：手动指定 MACA SDK 路径
-- ==========================================
option("maca-sdk")
    set_default("")
    set_showmenu(true)
    set_description("Path to MACA SDK root directory")
option_end()

-- ==========================================
-- Target: llaisys-device-mx
--
-- 编译 src/device/nvidia/*.cu，使用 mxcc 替代 nvcc
--
-- 关键: add_files("*.cu", {sourcekind = "cxx"})
--   将 .cu 文件强制标记为 C++ 源文件，绕开 xmake 的 CUDA 语言流程
--   （避免触发 find_cuda → "Cuda SDK not found!" 错误）
--   mxcc 作为 C++ 编译器，本身就能编译 .cu 中的 CUDA 语法
-- ==========================================
target("llaisys-device-mx")
    set_kind("shared")
    set_toolchains("mxcc")

    set_languages("cxx17")
    set_warnings("all", "error")
    set_encodings("none") -- 禁用全局的 utf-8 编码设置，避免 xmake 自动注入 -finput-charset=UTF-8
    set_strip("none")       -- 禁用 release 模式下默认产生的 -s (strip) 参数

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end

    -- 如果 mxcc 需要 "-x maca" 或类似参数来启用 MACA/CUDA 模式，在这里添加：
    add_cxflags("-x maca", {force = true})

    -- wcuda* symbols (including __wcuda_version_internal__) are provided by libruntime_cu.so
    add_links("runtime_cu")

    -- 【核心】将 .cu 文件强制当作 C++ 编译，避免触发 CUDA SDK 检测
    add_files("../src/device/nvidia/*.cu", {sourcekind = "cxx"})

    on_install(function (target) end)
target_end()

-- ==========================================
-- Target: llaisys-ops-mx
--
-- 编译 src/ops/*/nvidia/*.cu，使用 mxcc 替代 nvcc
-- ==========================================
target("llaisys-ops-mx")
    set_kind("shared")
    set_toolchains("mxcc")

    set_languages("cxx17")
    set_warnings("all", "error")
    set_encodings("none") -- 禁用全局的 utf-8 编码设置，避免 xmake 自动注入 -finput-charset=UTF-8
    set_strip("none")       -- 禁用 release 模式下默认产生的 -s (strip) 参数


    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end

    -- 如果 mxcc 需要 "-x maca" 或类似参数来启用 MACA/CUDA 模式，在这里添加：
    add_cxflags("-x maca", {force = true})

    -- Keep explicit dependency to avoid transitive wcuda symbol resolution issues.
    add_links("runtime_cu")

    add_deps("llaisys-tensor")
    -- Link against MetaX BLAS runtime for GEMM symbols (e.g. mcblasGemmEx)
    add_links("mcblas")

    -- 【核心】将 .cu 文件强制当作 C++ 编译
    add_files("../src/ops/*/nvidia/*.cu", {sourcekind = "cxx"})

    on_install(function (target) end)
target_end()