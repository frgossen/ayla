"""Compare driver-compiled SASS, resources, launch geometry, and TMA signature."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

def fingerprint(kernel, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    lib=ct.CDLL('libcuda.so.1')
    # Explicit sm_90a target: context-inferred sm_90 cannot link WGMMA code.
    error_log=ct.create_string_buffer(4096)
    option_ids=[13,9,5,6]  # line info; target; error log and capacity
    option_values=[1,0x10000+90,ct.addressof(error_log),len(error_log)]
    if kernel.ptx_optimization_level is not None:
        option_ids.append(7);option_values.append(kernel.ptx_optimization_level)
    count=len(option_ids)
    options=(ct.c_int*count)(*option_ids)
    values=(ct.c_void_p*count)(*option_values)
    state=ct.c_void_p()
    lib.cuLinkCreate_v2.argtypes=[ct.c_uint,ct.POINTER(ct.c_int),ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p)]
    lib.cuLinkAddData_v2.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_size_t,ct.c_char_p,
                                 ct.c_uint,ct.POINTER(ct.c_int),ct.POINTER(ct.c_void_p)]
    lib.cuLinkComplete.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_size_t)]
    lib.cuLinkDestroy.argtypes=[ct.c_void_p]
    def check(code):
        if code:raise RuntimeError(f'CUDA driver linker returned {code}: {error_log.value.decode(errors="replace")}')
    check(lib.cuLinkCreate_v2(count,options,values,ct.byref(state)))
    try:
        ptx=ct.create_string_buffer(kernel.ptx.encode())
        check(lib.cuLinkAddData_v2(state,1,ptx,len(ptx),'addmm.ptx'.encode(),0,None,None))
        cubin=ct.c_void_p();size=ct.c_size_t()
        check(lib.cuLinkComplete(state,ct.byref(cubin),ct.byref(size)))
        path.write_bytes(ct.string_at(cubin,size.value))
    finally:check(lib.cuLinkDestroy(state))
    output=subprocess.run(['/usr/local/cuda-13.0/bin/cuobjdump','--dump-sass','--dump-resource-usage',str(path)],
                          check=True,capture_output=True,text=True).stdout
    # cuobjdump prints the file path in an otherwise nonsemantic ELF header.
    normalized='\n'.join(line.strip() for line in output.replace(str(path),'<cubin>').splitlines()
                         if not line.strip().startswith(('File name:', 'Fatbin elf code:')))
    path.with_suffix('.sass').write_text(output)
    launch=dict(grid_dim=kernel.grid_dim,block_dim=kernel.block_dim,cluster_dim=kernel.cluster_dim,
                shared_mem_bytes=kernel.shared_mem_bytes,signature=[repr(t) for t in kernel.signature])
    digest=hashlib.sha256((normalized+'\n'+json.dumps(launch,sort_keys=True)).encode()).hexdigest()
    return dict(executable_fingerprint=digest,cubin_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                **launch)
