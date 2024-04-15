import taichi as ti
from taichi.lang.enums import AutodiffMode

ti.init(debug=True)

N = 16

x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
b = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def func_broke_rule_1():
    # 错误: 违反规则 #1, 修改完成之前读取全局区数据。
    loss[None] = x[1] * b[None]
    b[None] += 100


@ti.kernel
def func_equivalent():
    loss[None] = x[1] * 10


for i in range(N):
    x[i] = i
b[None] = 10
loss.grad[None] = 1

# with ti.ad.Tape(loss, validation=True):
#     func_broke_rule_1()
func_broke_rule_1.grad.autodiff_mode = AutodiffMode.VALIDATION
# func_broke_rule_1()
func_broke_rule_1.grad()
# Call func_equivalent to see the correct result
# with ti.ad.Tape(loss):
# func_equivalent()

assert x.grad[1] == 10.0
