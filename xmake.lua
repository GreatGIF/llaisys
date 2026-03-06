add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("openmp")
    set_default(false)
    set_showmenu(true)
    set_description("Enable OpenMP parallelization for CPU operators")
option_end()

option("cpu-blas")
    set_default(false)
    set_showmenu(true)
    set_description("Enable BLAS backend (OpenBLAS) for CPU linear operator")
option_end()

option("cpu-onednn")
    set_default(false)
    set_showmenu(true)
    set_description("Enable oneDNN backend for CPU linear operator")
option_end()

option("cpu-mkl")
    set_default(false)
    set_showmenu(true)
    set_description("Enable Intel MKL backend for CPU linear operator")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

if has_config("openmp") then
    add_defines("ENABLE_OPENMP")
    if not is_plat("windows") then
        if is_plat("linux") then
            add_cxflags("-fopenmp")
            local sys_libgomp = "/lib/x86_64-linux-gnu/libgomp.so.1"
            if os.isfile(sys_libgomp) then
                add_ldflags(sys_libgomp, {force = true})
                add_shflags(sys_libgomp, {force = true})
            else
                add_ldflags("-fopenmp")
                add_shflags("-fopenmp")
            end
        else
            add_cxflags("-fopenmp")
            add_ldflags("-fopenmp")
            add_shflags("-fopenmp")
        end
    end
end

if has_config("cpu-blas") then
    add_defines("ENABLE_CPU_BLAS")
    if is_plat("linux") then
        add_links("openblas")
    end
end

if has_config("cpu-onednn") then
    add_defines("ENABLE_CPU_ONEDNN")
    if is_plat("linux") then
        add_links("dnnl")
    end
end

if has_config("cpu-mkl") then
    add_defines("ENABLE_CPU_MKL")
    -- add_defines("MKL_HAS_HGEMM") -- 本地MKL版本不支持HGEMM
    if is_plat("linux") then
        -- Common header layout: /usr/include/mkl/mkl.h
        if os.isdir("/usr/include/mkl") then
            add_includedirs("/usr/include/mkl", {public = true, system = true})
        end

        -- oneAPI default include path
        local oneapi_mkl_include = "/opt/intel/oneapi/mkl/latest/include"
        if os.isdir(oneapi_mkl_include) then
            add_includedirs(oneapi_mkl_include, {public = true, system = true})
        end

        -- oneAPI default runtime library path
        local oneapi_mkl_libdir = "/opt/intel/oneapi/mkl/latest/lib/intel64"
        if os.isdir(oneapi_mkl_libdir) then
            add_linkdirs(oneapi_mkl_libdir)
            add_rpathdirs(oneapi_mkl_libdir)
        end

        add_links("mkl_rt")
    end
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-models")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/models/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    add_deps("llaisys-models")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/**.cc")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()