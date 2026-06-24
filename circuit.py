import pennylane as qml

def add_layer_pennylane(params, num_wires, noise_level=0.0):
    idx = 0
    for wire in range(num_wires - 1):
        p1 = params[idx]
        p2 = params[idx + 1]
        p3 = params[idx + 2]
        qml.U3(p1, p2, p3, wires=wire)
        idx += 3

        p_phase = params[idx]
        qml.ControlledPhaseShift(p_phase, wires=[wire + 1, wire])
        idx += 1

    if noise_level > 0.0:
        for dep_wire in range(num_wires):
            qml.DepolarizingChannel(noise_level, wires=dep_wire)

def create_qnode(num_qubits, num_layers, dev, noise_level=0.0):
    total_wires = num_qubits + 1
    params_per_layer = num_qubits * 4
    
    @qml.qnode(dev, interface="torch", diff_method="best", shots=1024)
    def circuit(weights):
        for wire in range(total_wires):
            qml.Hadamard(wires=wire)
        for layer_idx in range(num_layers):
            start = layer_idx * params_per_layer
            end = start + params_per_layer
            add_layer_pennylane(weights[ start:end], total_wires, noise_level)
        for wire in range(total_wires):
            qml.Hadamard(wires=wire)
        return qml.probs(wires=range(total_wires))

    return circuit