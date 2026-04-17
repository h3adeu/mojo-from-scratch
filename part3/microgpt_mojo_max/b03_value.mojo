# B3: Scalar autograd tape (microgpt.py Value), topology-order backward.
# Stored as structure-of-arrays so nodes stay in List[...] without custom copy/move.

from std.math import exp, log


comptime NodeId = Int


struct Tape:
    var data: List[Float64]
    var grad: List[Float64]
    var children: List[List[Int]]
    var local_grads: List[List[Float64]]

    def __init__(out self):
        self.data = List[Float64]()
        self.grad = List[Float64]()
        self.children = List[List[Int]]()
        self.local_grads = List[List[Float64]]()

    def _push_node(
        mut self, value: Float64, ch: List[Int], lg: List[Float64]
    ) -> NodeId:
        self.data.append(value)
        self.grad.append(0.0)
        self.children.append(ch.copy())
        self.local_grads.append(lg.copy())
        return len(self.data) - 1

    def node_data(self, i: NodeId) -> Float64:
        return self.data[i]

    def grad_at(self, i: NodeId) -> Float64:
        return self.grad[i]

    def set_data(mut self, i: NodeId, v: Float64):
        self.data[i] = v

    def set_grad(mut self, i: NodeId, g: Float64):
        self.grad[i] = g

    def add_grad(mut self, i: NodeId, d: Float64):
        self.grad[i] += d

    def clear_grads(mut self):
        for i in range(len(self.grad)):
            self.grad[i] = 0.0

    def leaf(mut self, value: Float64) -> NodeId:
        var ch = List[Int]()
        var lg = List[Float64]()
        return self._push_node(value, ch^, lg^)

    def add(mut self, a: NodeId, b: NodeId) -> NodeId:
        var ch = List[Int]()
        ch.append(a)
        ch.append(b)
        var lg = List[Float64]()
        lg.append(1.0)
        lg.append(1.0)
        var s = self.data[a] + self.data[b]
        return self._push_node(s, ch^, lg^)

    def mul(mut self, a: NodeId, b: NodeId) -> NodeId:
        var ch = List[Int]()
        ch.append(a)
        ch.append(b)
        var lg = List[Float64]()
        lg.append(self.data[b])
        lg.append(self.data[a])
        var p = self.data[a] * self.data[b]
        return self._push_node(p, ch^, lg^)

    def neg(mut self, a: NodeId) -> NodeId:
        var ch = List[Int]()
        ch.append(a)
        var lg = List[Float64]()
        lg.append(-1.0)
        return self._push_node(-self.data[a], ch^, lg^)

    def sub(mut self, a: NodeId, b: NodeId) -> NodeId:
        return self.add(a, self.neg(b))

    def pow_const(mut self, a: NodeId, expv: Float64) -> NodeId:
        var ad = self.data[a]
        var y = ad**expv
        var lg = expv * ad ** (expv - 1.0)
        var ch = List[Int]()
        ch.append(a)
        var lgs = List[Float64]()
        lgs.append(lg)
        return self._push_node(y, ch^, lgs^)

    def div(mut self, a: NodeId, b: NodeId) -> NodeId:
        var invb = self.pow_const(b, -1.0)
        return self.mul(a, invb)

    def log_(mut self, a: NodeId) -> NodeId:
        var ad = self.data[a]
        var ch = List[Int]()
        ch.append(a)
        var lg = List[Float64]()
        lg.append(1.0 / ad)
        return self._push_node(log(ad), ch^, lg^)

    def exp_(mut self, a: NodeId) -> NodeId:
        var ad = self.data[a]
        var y = exp(ad)
        var ch = List[Int]()
        ch.append(a)
        var lg = List[Float64]()
        lg.append(y)
        return self._push_node(y, ch^, lg^)

    def relu(mut self, a: NodeId) -> NodeId:
        var ad = self.data[a]
        var y = ad if ad > 0.0 else 0.0
        var g = 1.0 if ad > 0.0 else 0.0
        var ch = List[Int]()
        ch.append(a)
        var lg = List[Float64]()
        lg.append(g)
        return self._push_node(y, ch^, lg^)

    def sum_nodes(mut self, xs: List[Int]) -> NodeId:
        if len(xs) == 0:
            return self.leaf(0.0)
        var s = xs[0]
        for i in range(1, len(xs)):
            s = self.add(s, xs[i])
        return s

    def scale(mut self, a: NodeId, s: Float64) -> NodeId:
        return self.mul(a, self.leaf(s))


def _build_topo(mut tape: Tape, v: NodeId, mut topo: List[Int], mut visited: List[Bool]):
    if visited[v]:
        return
    visited[v] = True
    var ch = tape.children[v].copy()
    for i in range(len(ch)):
        _build_topo(tape, ch[i], topo, visited)
    topo.append(v)


def backward(mut tape: Tape, root: NodeId):
    tape.clear_grads()
    var visited = List[Bool]()
    for _ in range(len(tape.data)):
        visited.append(False)
    var topo = List[Int]()
    _build_topo(tape, root, topo, visited)
    tape.set_grad(root, 1.0)
    var ti = len(topo)
    while ti > 0:
        ti -= 1
        var v = topo[ti]
        var gv = tape.grad[v]
        var ch = tape.children[v].copy()
        var lgs = tape.local_grads[v].copy()
        for k in range(len(ch)):
            var c = ch[k]
            var lg = lgs[k]
            tape.add_grad(c, lg * gv)
