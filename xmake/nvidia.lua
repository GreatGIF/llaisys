target("llaisys-device-nvidia")

    set_kind("shared")
    -- set_kind("static")
    -- set_policy("build.cuda.devlink", true) -- 非 Cuda binary/shared 依赖 Cuda static 目标的情况需要开启 devlink
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler -fPIC")
        add_culdflags("-Xcompiler -fPIC")
    end

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("../src/device/nvidia/*.cu")
    add_cugencodes("native")
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")

    set_kind("shared")
    -- set_kind("static")
    -- set_policy("build.cuda.devlink", true)
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler -fPIC")
        add_culdflags("-Xcompiler -fPIC")
    end

    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("../src/ops/*/nvidia/*.cu")
    add_cugencodes("native")

    on_install(function (target) end)
target_end()

