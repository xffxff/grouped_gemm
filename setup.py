import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

use_jiangs_code = True

if use_jiangs_code:

#########################################################################
#
# jiangs grouped gemm
#
#########################################################################

    import subprocess
    import re
    import shutil
    import os

    def found_cmake() -> bool:
        """"Check if valid CMake is available

        CMake 3.18 or newer is required.

        """

        # Check if CMake is available
        try:
            _cmake_bin = cmake_bin()
        except FileNotFoundError:
            return False

        # Query CMake for version info
        output = subprocess.run(
            [_cmake_bin, "--version"],
            capture_output=True,
            check=True,
            universal_newlines=True,
        )
        match = re.search(r"version\s*([\d.]+)", output.stdout)
        version = match.group(1).split('.')
        version = tuple(int(v) for v in version)
        return version >= (3, 18)

    def cmake_bin() -> Path:
        """Get CMake executable

        Throws FileNotFoundError if not found.

        """

        # Search in CMake Python package
        _cmake_bin = None
        try:
            import cmake
        except ImportError:
            pass
        else:
            cmake_dir = Path(cmake.__file__).resolve().parent
            _cmake_bin = cmake_dir / "data" / "bin" / "cmake"
            if not _cmake_bin.is_file():
                _cmake_bin = None

        # Search in path
        if _cmake_bin is None:
            _cmake_bin = shutil.which("cmake")
            if _cmake_bin is not None:
                _cmake_bin = Path(_cmake_bin).resolve()

        # Return executable if found
        if _cmake_bin is None:
            raise FileNotFoundError("Could not find CMake executable")
        return _cmake_bin


    setup_requires = []

    if not found_cmake():
        setup_requires.append("cmake>=3.18")

    build_dir = os.path.dirname(os.path.abspath(__file__)) + '/grouped_gemm_dev/build'
    cmake_dir = os.path.dirname(os.path.abspath(__file__)) + '/grouped_gemm_dev'

    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        print(f"Directory '{build_dir}' created.")
    else:
        print(f"Directory '{build_dir}' already exists.")

    _cmake_bin = str(cmake_bin())
    subprocess.run([_cmake_bin, "-S", cmake_dir, "-B", build_dir])
    subprocess.run([_cmake_bin, "--build", build_dir, "--parallel"])


    setup(
        name="grouped_gemm_01",
        version="0.5",
        author="Jiang Shao, Shiqing Fan",
        author_email="jiangs@nvidia.com, shiqingf@nvidia.com",
        description="GroupedGEMM CUDA Extension of Pytorch for MoE",
        url="https://github.com/fanshiqing/grouped_gemm@releases/nv_grouped_gemm_v0.5",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: BSD License",
            "Operating System :: Unix",
        ],
        package_dir={'grouped_gemm_01': 'grouped_gemm_dev'},
        packages=['grouped_gemm_01', 'grouped_gemm_01.moe'],
        package_data={
        'grouped_gemm_01': ['build/libmoe_unit_ops.so'],
        },
        cmdclass={"build_ext": BuildExtension},
        install_requires=["absl-py", "numpy", "torch"],
    )

else:

#########################################################################
#
# tgale96 grouped gemm
#
#########################################################################

    if not torch.cuda.is_available():
        if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"


    cwd = Path(os.path.dirname(os.path.abspath(__file__)))
    _dc = torch.cuda.get_device_capability()
    if _dc[0] < 8:
        print("Unsupported compute capability, only device capability >=80 are supported.")
        # DEBUG: set dc=8 as a workaround when ci is scheduled on V100.
        _dc = (8, 0)
        # sys.exit(0)

    _dc = f"{_dc[0]}{_dc[1]}"

    ext_modules = [
        CUDAExtension(
            "grouped_gemm_backend",
            ["csrc/ops.cu", "csrc/grouped_gemm.cu", "csrc/sinkhorn.cu"],
            include_dirs = [
                f"{cwd}/third_party/cutlass/include/"
            ],
            extra_compile_args={
                "cxx": [
                    "-fopenmp", "-fPIC", "-Wno-strict-aliasing"
                ],
                "nvcc": [
                    f"--generate-code=arch=compute_{_dc},code=sm_{_dc}",
                    f"-DGROUPED_GEMM_DEVICE_CAPABILITY={_dc}",
                    # NOTE: CUTLASS requires c++17.
                    "-std=c++17",
                ],
            }
        )
    ]

    setup(
        name="grouped_gemm",
        version="0.0.1",
        author="Trevor Gale, Shiqing Fan",
        author_email="tgale@stanford.edu, shiqingf@nvidia.com",
        description="GEMM Grouped",
        url="https://github.com/fanshiqing/grouped_gemm",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: BSD License",
            "Operating System :: Unix",
        ],
        packages=find_packages(),
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        install_requires=["absl-py", "numpy", "torch"],
    )
