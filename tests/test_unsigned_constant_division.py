"""Execute constant division/remainder on adversarial full-range uint32 inputs."""

import random

import pytest

from ayla.base.ir import Block, Value
from ayla.gpu_dialect import (
    F32Type,
    GmemTensorType,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexType,
    KernelOp,
    ModuleOp,
    PTXTemplateOp,
)
from ayla.ptx_codegen import make_ptx_kernel


def arithmetic_kernel(divisor, count):
    source = Value(name="source", type=GmemTensorType.build([count], F32Type()))
    output = Value(name="output", type=GmemTensorType.build([2 * count], F32Type()))
    read = PTXTemplateOp.build(
        [source],
        [IndexType()],
        operand_names=["source"],
        result_names=["value"],
        template="""mov.u32 ${b32:tid}, %tid.x;
mov.u32 ${b32:block}, %ctaid.x;
mad.lo.u32 ${b32:idx}, ${b32:block}, 256, ${b32:tid};
mul.wide.u32 ${b64:offset}, ${b32:idx}, 4;
add.u64 ${b64:ptr}, ${b64:operand.source.ptr}, ${b64:offset};
ld.global.u32 ${b32:result.value.val}, [${b64:ptr}];
""",
    )
    constant = IndexConstantOp.build(divisor)
    divide = IndexDivOp.build(read.result(), constant.result())
    modulo = IndexModOp.build(read.result(), constant.result())
    write = PTXTemplateOp.build(
        [output, divide.result(), modulo.result()],
        [],
        operand_names=["output", "quotient", "remainder"],
        template=f"""mov.u32 ${{b32:tid}}, %tid.x;
mov.u32 ${{b32:block}}, %ctaid.x;
mad.lo.u32 ${{b32:idx}}, ${{b32:block}}, 256, ${{b32:tid}};
mul.wide.u32 ${{b64:offset}}, ${{b32:idx}}, 4;
add.u64 ${{b64:ptr}}, ${{b64:operand.output.ptr}}, ${{b64:offset}};
st.global.u32 [${{b64:ptr}}], ${{b32:operand.quotient.val}};
add.u64 ${{b64:ptr}}, ${{b64:ptr}}, {count * 4};
st.global.u32 [${{b64:ptr}}], ${{b32:operand.remainder.val}};
""",
    )
    kernel = KernelOp.build(
        "constant_division",
        Block(arguments=[source, output], ops=[read, constant, divide, modulo, write]),
        grid_dim=(count // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    kernel.attrs["integer_index_division"] = True
    return make_ptx_kernel(
        ModuleOp.build(Block(ops=[kernel]), attrs={"target": "sm_100a"})
    )


@pytest.mark.parametrize(
    "divisor", [3, 6, 7, 10, 12, 14, 31, 65535, 2**31 - 1, 2**32 - 5]
)
def test_full_range_unsigned_constant_division_gpu(divisor):
    import torch
    from ayla.ptx_runtime import make_ptx_launcher

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    rng = random.Random(divisor)
    values = [0, 1, 2**31 - 1, 2**31, 2**32 - 1]
    for quotient in [0, 1, 2, 3, 2**32 // divisor - 1, 2**32 // divisor]:
        values.extend(
            v for delta in [-1, 0, 1] if 0 <= (v := quotient * divisor + delta) < 2**32
        )
    values.extend(rng.randrange(2**32) for _ in range(16384 - len(values)))
    source = torch.tensor(values, dtype=torch.int64, device="cuda").to(torch.int32)
    output = torch.empty((2, len(values)), dtype=torch.int32, device="cuda")
    kernel = arithmetic_kernel(divisor, len(values))
    make_ptx_launcher(kernel)(source, output)
    expected = torch.tensor([values], dtype=torch.int64, device="cuda")
    assert torch.equal(
        output[0].to(torch.int64) & (2**32 - 1), (expected // divisor)[0]
    )
    assert torch.equal(output[1].to(torch.int64) & (2**32 - 1), (expected % divisor)[0])
